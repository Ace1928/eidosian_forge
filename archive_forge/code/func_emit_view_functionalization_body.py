from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def emit_view_functionalization_body(g: NativeFunctionsViewGroup, *, view_inplace: bool) -> str:
    if view_inplace:
        assert g.view_inplace is not None
        f = g.view_inplace
    else:
        f = g.view
    assert g.view_copy is not None
    with native_function_manager(f):
        call_sig = DispatcherSignature.from_schema(g.view_copy.func)
        api_name = g.view_copy.func.name.unambiguous_name()
        noop_api_name = f.func.name.unambiguous_name()
        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        assert_view_op_properties(f.func)
        view_tensor_name = dispatcher_sig.arguments()[0].name
        return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()
        unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(dispatcher_sig, is_view_op=True)
        view_redispatch_args = [e.expr for e in translate(unwrapped_args_ctx, call_sig.arguments(), method=False)]
        forward_lambda = FunctionalizationLambda.from_func(g, is_reverse=False)
        reverse_lambda = FunctionalizationLambda.from_func(g, is_reverse=True)
        meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
        meta_call_args = [e.expr for e in translate(meta_call_ctx, call_sig.arguments(), method=False)]
        if 'inplace_view' in f.tags:
            return f"\n    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{\n      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{\n        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.\n        {unwrap_tensor_args_str}\n        at::AutoDispatchSkipFunctionalize guard;\n        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});\n      }}\n      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();\n      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(\n        {forward_lambda.decl()} {{\n          if (reapply_views) {{\n            return {forward_lambda.inner_call(reapply_views=True)}\n          }} else {{\n            return {forward_lambda.inner_call(reapply_views=False)}\n          }}\n        }},\n        {reverse_lambda.decl()} {{\n          return {reverse_lambda.inner_call()}\n        }}\n      );\n      auto compute_reference_meta =\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);\n      {return_type} reference_tensor_output;\n      if (compute_reference_meta) {{\n        {meta_conversion_str}\n        at::AutoDispatchSkipFunctionalize func_guard;\n        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});\n      }}\n      // This function adds the above view meta to the current tensor and replays them off the base,\n      // mutating the size/stride info of the current FunctionalTensorWrapper.\n      // Because of this, we need to make sure to run the reference shape function above,\n      // BEFORE doing this (otherwise we'll end up runnin the reference function using the wrong sizes/strides)\n      at::functionalization::impl::mutate_view_meta({view_tensor_name}, view_meta);\n      // See  Note [Propagating strides in the functionalization pass]\n      // XLA/LTC don't implement the logic to propagate strides correctly, so we need to rely\n      // on a reference implementation here (instead of relying on the output from the forward lambda\n      // having the correct stride info)\n      if (compute_reference_meta) {{\n        at::functionalization::impl::set_sizes_strides_offset({view_tensor_name}, reference_tensor_output);\n      }}\n      return {view_tensor_name};\n    }}\n"
        else:
            is_multi_output_view = isinstance(f.func.returns[0].type, ListType)
            return f"\n    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{\n      {unwrap_tensor_args_str}\n      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{\n        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.\n        at::AutoDispatchSkipFunctionalize guard;\n        return at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});\n      }}\n      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();\n      auto compute_reference_meta =\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||\n        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);\n      {return_type} reference_tensor_output;\n      if (compute_reference_meta) {{\n        {meta_conversion_str}\n        at::AutoDispatchSkipFunctionalize func_guard;\n        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);\n        reference_tensor_output = at::_ops::{noop_api_name}::call({', '.join(meta_call_args)});\n      }}\n      {return_type} tmp_output;\n      {{\n        at::AutoDispatchSkipFunctionalize guard;\n        if (reapply_views) {{\n          tmp_output = at::_ops::{noop_api_name}::call({', '.join(view_redispatch_args)});\n        }} else {{\n          tmp_output = at::_ops::{api_name}::call({', '.join(view_redispatch_args)});\n        }}\n      }}\n      at::functionalization::ViewMeta view_meta = at::functionalization::ViewMeta(\n        {forward_lambda.decl()} {{\n          if (reapply_views) {{\n            return {forward_lambda.inner_call(reapply_views=True)}\n          }} else {{\n            return {forward_lambda.inner_call(reapply_views=False)}\n          }}\n        }},\n        {reverse_lambda.decl()} {{\n          return {reverse_lambda.inner_call()}\n        }},\n        /*is_multi_output=*/{str(is_multi_output_view).lower()}\n      );\n      auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, {view_tensor_name}, view_meta);\n      // See  Note [Propagating strides in the functionalization pass]\n      if (compute_reference_meta) {{\n        at::functionalization::impl::set_sizes_strides_offset(out, reference_tensor_output);\n      }}\n      return out;\n    }}\n"
import itertools
from contextlib import nullcontext
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import dynamo_timed, preserve_rng_state
from torch._guards import detect_fake_mode
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from . import config
from .partitioners import default_partition
from ._aot_autograd.utils import (  # noqa: F401
from ._aot_autograd.logging_utils import (  # noqa: F401
from ._aot_autograd.functional_utils import (  # noqa: F401
from ._aot_autograd.schemas import (  # noqa: F401
from ._aot_autograd.subclass_utils import (  # noqa: F401
from ._aot_autograd.collect_metadata_analysis import (  # noqa: F401
from ._aot_autograd.input_output_analysis import (  # noqa: F401
from ._aot_autograd.traced_function_transforms import (  # noqa: F401
from ._aot_autograd.runtime_wrappers import (  # noqa: F401
from ._aot_autograd.dispatch_and_compile_graph import (  # noqa: F401
from ._aot_autograd.jit_compile_runtime_wrappers import (  # noqa: F401
@dynamo_timed
def create_aot_dispatcher_function(flat_fn, flat_args: List[Any], aot_config: AOTConfig):
    """
    Traces the forward and backward graphs of the attr:`flat_fn` to generate a
    joint graph. The joint graph is an Fx graph with Aten ops. Please refer to
    the tracing mechanism to understand the graph capturing details.

    The joint graph is then passed through attr:`partition_fn` to isolate the
    forward and backward portions, which are then respectively compiled via the
    provided attr:`fw_compiler` and attr:`bw_compiler`.

    The resulting compiled forward and backward graphs are then wrapped up in a
    ``torch.autograd.Function`` object.

    The calling convention here is that the first aot_config.num_params_buffers
    inputs in flat_args are parameters and buffers, and the rest are inputs.

    We use this to assume that parameters/buffer's shapes don't change.

    Note: this function is used both by aot_function and aot_export (controlled by aot_config.is_export)
        When aot_config.is_export is True, we return an FX graph + metadata
        When aot_config.is_export is False, we return an ordinary runtime function
    """
    if aot_config.decompositions is None:
        aot_config.decompositions = {}
    aot_config.decompositions = {**aot_autograd_decompositions, **aot_config.decompositions}
    if config.functionalize_rng_ops:
        aot_config.decompositions = {**rng_decompositions, **aot_config.decompositions}
    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env
    python_dispatcher_mode = enable_python_dispatcher() if shape_env is not None else nullcontext()
    with torch.autograd.set_multithreading_enabled(False), preserve_rng_state(), fake_mode, python_dispatcher_mode, PhiloxStateTracker():

        def process_inputs(flat_args):

            def convert(idx, x):
                if shape_env is not None:
                    from torch._dynamo.source import ConstantSource
                    if isinstance(x, int):
                        if aot_config.is_export:
                            return x
                        source = ConstantSource(f'sym_{idx}')
                        return shape_env.create_symintnode(shape_env.create_symbol(x, source), hint=x, source=source)
                if not isinstance(x, torch.Tensor):
                    return x
                if isinstance(x, FakeTensor):
                    assert x.fake_mode is fake_mode
                    return x
                if is_traceable_wrapper_subclass(x):
                    attrs, _ = x.__tensor_flatten__()
                    if all((isinstance(getattr(x, attr), FakeTensor) for attr in attrs)):
                        assert all((getattr(x, attr).fake_mode is fake_mode for attr in attrs))
                        return x
                symbolic_context = None
                source = None
                if (tracing_context := torch._guards.TracingContext.try_get()):
                    if x in tracing_context.tensor_to_context:
                        symbolic_context = tracing_context.tensor_to_context[x]
                        source = symbolic_context.tensor_source
                if idx < aot_config.num_params_buffers and config.static_weight_shapes and (not symbolic_context):
                    return fake_mode.from_tensor(x, static_shapes=True)
                return fake_mode.from_tensor(x, static_shapes=False, symbolic_context=symbolic_context, source=source)
            return [convert(idx, x) for idx, x in enumerate(flat_args)]
        fake_flat_args = process_inputs(flat_args)
        needs_autograd = any((x.requires_grad for x in fake_flat_args if isinstance(x, Tensor))) and torch.is_grad_enabled()
        with enable_python_dispatcher():
            with patch('torch.cuda.set_rng_state', lambda *args: None):
                fw_metadata = run_functionalized_fw_and_collect_metadata(flat_fn, keep_input_mutations=aot_config.keep_inference_input_mutations, is_train=needs_autograd)(*fake_flat_args)
                req_subclass_dispatch = requires_subclass_dispatch(fake_flat_args, fw_metadata)
                if needs_autograd and (not any((x.requires_grad for x in fw_metadata.output_info))):
                    needs_autograd = False
                    if req_subclass_dispatch:
                        fw_metadata = run_functionalized_fw_and_collect_metadata(flat_fn, keep_input_mutations=aot_config.keep_inference_input_mutations and (not needs_autograd), is_train=needs_autograd)(*fake_flat_args)
                    else:
                        fw_metadata = ViewAndMutationMeta(input_info=fw_metadata.input_info, output_info=fw_metadata.output_info, num_intermediate_bases=fw_metadata.num_intermediate_bases, keep_input_mutations=aot_config.keep_inference_input_mutations and (not needs_autograd), traced_tangents=fw_metadata.traced_tangents, subclass_inp_meta=fw_metadata.subclass_inp_meta, subclass_fw_graph_out_meta=fw_metadata.subclass_fw_graph_out_meta, subclass_tangent_meta=fw_metadata.subclass_tangent_meta, is_train=needs_autograd)
        if fw_metadata.num_intermediate_bases > 0:
            assert not req_subclass_dispatch, f'torch.compile is currently being used with tensor subclass inputs:\n{','.join([str(type(x)) for x in fake_flat_args])}. We are attempting to a compile a graph with two graph outputs\nthat alias one another, which is currently unsupported in the subclass use case. If you run into this,\nplease file a github issue'
        if aot_config.is_export:
            if len([x for x in fw_metadata.input_info if x.mutates_metadata]) != 0:
                raise RuntimeError(f'Found an input that received a metadata mutation, through e.g. a call to `.resize_()` or `.transpose_()`.\nThis is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.\n\nfw_metadata={str(fw_metadata)}')
            if len([x for x in fw_metadata.input_info if x.requires_grad and x.mutates_data]) != 0:
                raise RuntimeError(f'Found a graph input that requires gradients, and received a mutation.\nThis is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.\n\nfw_metadata={str(fw_metadata)}')
            if req_subclass_dispatch:
                raise RuntimeError('aot_export is not currently supported with traceable tensor subclass.\nIf you need this feature, please comment on <CREATE_ISSUE_LINK>')
            if config.functionalize_rng_ops:
                raise RuntimeError('Functionalized RNG is not currently supported in the aot_export workflow. Please file a github issue,\nor otherwise set torch._functorch.config.functionalize_rng_ops = False.')
        if needs_autograd:
            compiler_fn = aot_dispatch_autograd_graph if aot_config.is_export else aot_dispatch_autograd
        else:
            compiler_fn = aot_dispatch_base_graph if aot_config.is_export else aot_dispatch_base
        compiler_fn = partial(aot_wrapper_synthetic_base, compiler_fn=compiler_fn, needs_autograd=needs_autograd)
        compiler_fn = partial(aot_wrapper_dedupe, compiler_fn=compiler_fn)
        compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
        if aot_config.is_export:
            mutated_user_inp_locs = [idx - aot_config.num_params_buffers for idx in fw_metadata.mutated_inp_runtime_indices if idx >= aot_config.num_params_buffers]
            if len(mutated_user_inp_locs) > 0:
                raise RuntimeError(f'\nFound following user inputs located at {mutated_user_inp_locs} are mutated. This is currently banned in the aot_export workflow.\nIf you need this functionality, please file a github issue.\n\nfw_metadata={str(fw_metadata)}')
            assert isinstance(compiled_fn, torch.fx.GraphModule)
            return (compiled_fn, fw_metadata)
        if not hasattr(compiled_fn, '_boxed_call'):
            compiled_fn = make_boxed_func(compiled_fn)
        return compiled_fn
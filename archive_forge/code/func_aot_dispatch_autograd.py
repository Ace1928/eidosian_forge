import logging
from contextlib import nullcontext
from functools import wraps
from typing import Any, List, Optional
import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._dynamo.utils import lazy_format_graph_code
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._logging import getArtifactLogger
from torch._prims_common import CUDARngStateHelper
from torch._subclasses import FakeTensor
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals
from .. import config
from .dispatch_and_compile_graph import (
from .logging_utils import describe_input, format_guard_bug_msg, track_graph_compiling
from .runtime_wrappers import (
from .schemas import (
from .subclass_utils import unwrap_tensor_subclasses, wrap_tensor_subclasses
from .utils import (
def aot_dispatch_autograd(flat_fn, flat_args: List[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    fx_g, joint_inputs, maybe_subclass_meta = aot_dispatch_autograd_graph(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
    traced_tangents = pytree.tree_map(lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x, fw_metadata.traced_tangents)
    disable_amp = torch._C._is_any_autocast_enabled()
    if aot_config.enable_log:
        aot_joint_log.info('%s', lazy_format_graph_code('Joint graph', fx_g, aot_config.aot_id))
    with torch.no_grad():
        inner_meta = fw_metadata if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
        with track_graph_compiling(aot_config, 'joint'):
            num_inner_fwd_outputs = inner_meta.num_mutated_inp_runtime_indices + inner_meta.num_outputs + inner_meta.num_intermediate_bases + inner_meta.num_outputs_rng_offset
            fw_module, bw_module = aot_config.partition_fn(fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs)
            fw_outs = next((n for n in fw_module.graph.nodes if n.op == 'output')).args[0]
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)
            symint_outs_saved_for_bw = [n for n in fw_outs_saved_for_bw if is_sym_node(n)]
            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            inner_meta.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            _num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
        _indices_of_inps_to_detach = []
        bw_outs = next((n for n in bw_module.graph.nodes if n.op == 'output')).args[0]
        if maybe_subclass_meta is None:
            assert len(bw_outs) == len(fw_metadata.input_info) + inner_meta.num_outputs_rng_offset
            for i, bw_out in enumerate(bw_outs):
                if bw_out is None:
                    _indices_of_inps_to_detach.append(i)
        if aot_config.enable_log:
            aot_graphs_log.info('%s', lazy_format_graph_code('Forward graph', fw_module, aot_config.aot_id))
            aot_graphs_log.info('%s', lazy_format_graph_code('Backward graph', bw_module, aot_config.aot_id))
        with track_graph_compiling(aot_config, 'forward'):
            adjusted_flat_args = joint_inputs[0]
            if config.functionalize_rng_ops:
                fake_mode = detect_fake_mode()
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
                adjusted_flat_args.extend([seed, offset])
            if (tracing_context := torch._guards.TracingContext.try_get()):
                tracing_context.fw_metadata = inner_meta
            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)
            if not hasattr(compiled_fw_func, '_boxed_call'):
                compiled_fw_func = make_boxed_func(compiled_fw_func)
            if maybe_subclass_meta is not None:
                compiled_fw_func = aot_dispatch_subclass_wrapper(compiled_fw_func, subclass_metas=fw_metadata.subclass_fw_graph_out_meta, num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw)
                if not hasattr(compiled_fw_func, '_boxed_call'):
                    compiled_fw_func = make_boxed_func(compiled_fw_func)
        with track_graph_compiling(aot_config, 'backward'):
            placeholder_list = fx_placeholder_vals(bw_module)
            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                forward_saved_for_backwards_strides = fwd_output_strides[inner_meta.tensors_saved_for_backwards_slice]
            for i in range(len(placeholder_list)):
                ph_arg = placeholder_list[i]
                if not isinstance(ph_arg, torch.Tensor):
                    continue
                if forward_saved_for_backwards_strides is None:
                    continue
                real_stride = None
                j = i - len(symint_outs_saved_for_bw)
                if 0 <= j < len(forward_saved_for_backwards_strides):
                    real_stride = forward_saved_for_backwards_strides[j]
                if real_stride is None:
                    continue
                if _get_symint_hints(ph_arg.stride()) != real_stride:
                    placeholder_list[i] = ph_arg.as_strided(ph_arg.size(), real_stride)
            compiled_bw_func = None
            if len(symint_outs_saved_for_bw):
                context = torch._C._DisableAutocast if disable_amp else nullcontext
                with context():
                    try:
                        compiled_bw_func = aot_config.bw_compiler(bw_module, placeholder_list)
                    except Exception:
                        log.warning('failed to eagerly compile backwards for dynamic, suppressing in case backwards not needed', exc_info=True)
    saved_context = TracingContext.try_get()

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = compiled_bw_func
        metadata: ViewAndMutationMeta = fw_metadata
        maybe_subclass_metadata: Optional[SubclassMeta] = maybe_subclass_meta
        num_symints_saved_for_bw = _num_symints_saved_for_bw

        @staticmethod
        def _compiled_autograd_key(ctx):
            return (aot_config.aot_id, *ctx.symints)

        @staticmethod
        def forward(ctx, *deduped_flat_tensor_args):
            args = deduped_flat_tensor_args
            marked_dirty_inps = []
            for i in fw_metadata.mutated_graph_handled_indices:
                ctx.mark_dirty(deduped_flat_tensor_args[i])
                marked_dirty_inps.append(deduped_flat_tensor_args[i])
            if CompiledFunction.metadata.is_rng_op_functionalized:
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                args = (*args, seed, offset)
            fw_outs = call_func_at_runtime_with_args(CompiledFunction.compiled_fw, args, disable_amp=disable_amp)
            num_outputs = CompiledFunction.metadata.num_outputs
            num_outputs_aliased = CompiledFunction.metadata.num_outputs_aliased
            num_intermediate_bases = CompiledFunction.metadata.num_intermediate_bases
            num_symints_saved_for_bw = CompiledFunction.num_symints_saved_for_bw
            num_mutated_runtime_inps = CompiledFunction.metadata.num_mutated_inp_runtime_indices
            num_forward_returns = CompiledFunction.metadata.num_forward_returns
            num_forward = CompiledFunction.metadata.num_forward
            tensors_saved_for_backwards = fw_outs[CompiledFunction.metadata.tensors_saved_for_backwards_slice]
            assert all((isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards))
            ctx.save_for_backward(*(x.detach() if x._is_view() else x for x in tensors_saved_for_backwards))
            symint_outs = fw_outs[CompiledFunction.metadata.symints_saved_for_backwards_slice]
            assert all((isinstance(x, (int, float, torch.SymInt, torch.SymFloat)) for x in symint_outs)), str([type(x) for x in symint_outs])
            ctx.symints = symint_outs
            raw_returns = fw_outs[0:num_forward_returns]
            if num_mutated_runtime_inps > 0:
                for i, idx in enumerate(CompiledFunction.metadata.mutated_inp_runtime_indices):
                    info = CompiledFunction.metadata.input_info[idx]
                    if info.mutates_metadata and (not info.mutates_data):
                        raw_returns[i] = TensorAlias(raw_returns[i])
                if config.debug_assert:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_runtime_inps]
                    mut_inp_infos = [x for x in CompiledFunction.metadata.input_info if x.mutates_data or x.mutates_metadata]
                    assert len(user_mutated_inputs_raw) == len(mut_inp_infos)
            if CompiledFunction.metadata.num_unsafe_view_outputs > 0:
                for idx in CompiledFunction.metadata.unsafe_view_out_indices:
                    raw_return_idx = num_mutated_runtime_inps + idx
                    o = raw_returns[raw_return_idx]
                    raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(o, o.shape)
            if num_outputs_aliased > 0:
                for idx in CompiledFunction.metadata.aliased_out_indices:
                    raw_return_idx = num_mutated_runtime_inps + idx
                    raw_returns[raw_return_idx] = TensorAlias(raw_returns[raw_return_idx])
                if config.debug_assert:
                    intermediates_raw = raw_returns[num_mutated_runtime_inps + num_outputs:]
                    assert not any((isinstance(x, TensorAlias) for x in intermediates_raw))
            raw_returns_not_including_intermediate_bases = raw_returns[:num_mutated_runtime_inps + num_outputs]
            raw_returns_meta = [x for x in CompiledFunction.metadata.input_info if x.mutation_type == MutationType.MUTATED_OUT_GRAPH] + CompiledFunction.metadata.output_info
            fw_outs_not_requiring_grad = [x for i, x in enumerate(raw_returns_not_including_intermediate_bases) if isinstance(x, torch.Tensor) and (not raw_returns_meta[i].requires_grad)]
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
            ctx._materialize_non_diff_grads = False
            functionalized_rng_runtime_epilogue(CompiledFunction.metadata, fw_outs[num_forward_returns:num_forward], return_new_outs=False)
            return tuple(raw_returns) + tuple(marked_dirty_inps)

        @staticmethod
        def backward(ctx, *flat_args):
            num_intermediate_bases = CompiledFunction.metadata.num_intermediate_bases
            num_graph_handled_inputs = CompiledFunction.metadata.num_mutated_graph_handled_indices
            num_mutated_runtime_inps = CompiledFunction.metadata.num_mutated_inp_runtime_indices
            expected_grad_outs = CompiledFunction.metadata.num_outputs + num_mutated_runtime_inps + num_intermediate_bases
            if num_graph_handled_inputs > 0:
                flat_args = flat_args[:-num_graph_handled_inputs]
            assert len(flat_args) == expected_grad_outs
            out_info = CompiledFunction.metadata.output_info
            inp_tangents, out_tangents, intermediate_base_tangents = (flat_args[0:num_mutated_runtime_inps], flat_args[num_mutated_runtime_inps:num_mutated_runtime_inps + CompiledFunction.metadata.num_outputs], flat_args[num_mutated_runtime_inps + CompiledFunction.metadata.num_outputs:])
            input_info = CompiledFunction.metadata.input_info
            inp_tangents_filtered = [x for x, info_idx in zip(inp_tangents, CompiledFunction.metadata.mutated_inp_runtime_indices) if input_info[info_idx].mutates_data and input_info[info_idx].requires_grad]
            out_tangents_filtered = [x for x, info in zip(out_tangents, out_info) if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view] and issubclass(info.raw_type, torch.Tensor) and info.requires_grad]
            flat_bw_args_with_grads = [*inp_tangents_filtered, *out_tangents_filtered, *intermediate_base_tangents]
            num_flat_bw_args_with_grads = len(flat_bw_args_with_grads)
            rng_args = []
            if CompiledFunction.metadata.is_rng_op_functionalized:
                rng_args = CUDARngStateHelper.get_torch_state_as_tuple()
            all_args = [*ctx.symints, *ctx.saved_tensors, *flat_bw_args_with_grads, *rng_args]
            del flat_bw_args_with_grads
            tangents_start_idx = len(all_args) - num_flat_bw_args_with_grads - len(rng_args)
            tangents_end_idx = len(all_args) - len(rng_args)
            assert len(CompiledFunction.metadata.output_types) == num_flat_bw_args_with_grads
            grad_output_types = [type(x) for x in all_args[-num_flat_bw_args_with_grads:]]
            grad_output_types_ = [torch.Tensor if x is FakeTensor else x for x in grad_output_types]
            assert grad_output_types_ == CompiledFunction.metadata.output_types, f'We incorrectly attempted to compile the backward with incorrect subclass metadata.\nIf you run into this error, please file an issue.\nExpected grad_output types: {str(CompiledFunction.metadata.output_types)}\nGot grad_output types: {str(grad_output_types)}'
            if CompiledFunction.maybe_subclass_metadata is not None:
                len_tangents = len(unwrap_tensor_subclasses(all_args[tangents_start_idx:tangents_end_idx], is_joint_structure=False))
                all_args = unwrap_tensor_subclasses(all_args, is_joint_structure=False)
                tangents_start_idx = len(all_args) - len_tangents - len(rng_args)
                tangents_end_idx = tangents_start_idx + len_tangents
            all_args = [t.contiguous() if tangents_start_idx <= i < tangents_end_idx else t for i, t in enumerate(all_args)]

            def call_compiled_backward():
                if ctx._is_compiled_autograd_tracing():
                    symints = ctx._get_compiled_autograd_symints()
                    assert len(symints) == len(ctx.symints)
                    all_args[:len(symints)] = symints
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with context():
                        out = normalize_as_list(bw_module(*all_args))
                    out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
                    return tuple(out)
                ctx.maybe_clear_saved_tensors()
                if CompiledFunction.compiled_bw is None:
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with tracing(saved_context), context(), track_graph_compiling(aot_config, 'backward'):
                        CompiledFunction.compiled_bw = aot_config.bw_compiler(bw_module, placeholder_list)
                out = call_func_at_runtime_with_args(CompiledFunction.compiled_bw, all_args, steal_args=True, disable_amp=disable_amp)
                out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
                return tuple(out)
            if torch.is_grad_enabled() and any((t.requires_grad for t in all_args if isinstance(t, torch.Tensor))):

                class CompiledFunctionBackward(torch.autograd.Function):

                    @staticmethod
                    def forward(ctx, *unused_args):
                        outs = call_compiled_backward()
                        if CompiledFunction.maybe_subclass_metadata is not None:
                            assert CompiledFunction.maybe_subclass_metadata.grad_input_metas is not None
                            outs_wrapped = wrap_tensor_subclasses(outs, subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas)
                            return outs_wrapped
                        return outs

                    @staticmethod
                    def backward(ctx, *args):
                        raise RuntimeError('torch.compile with aot_autograd does not currently support double backward')
                CompiledFunctionBackward._compiled_autograd_key = CompiledFunction._compiled_autograd_key
                out = CompiledFunctionBackward.apply(*all_args)
            else:
                out = call_compiled_backward()
            if CompiledFunction.maybe_subclass_metadata is not None:
                assert CompiledFunction.maybe_subclass_metadata.grad_input_metas is not None
                outs_wrapped = wrap_tensor_subclasses(out, subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas)
                return outs_wrapped
            return out
    compiled_function = create_runtime_wrapper(CompiledFunction.apply, runtime_metadata=fw_metadata, indices_of_inps_to_detach=_indices_of_inps_to_detach, trace_joint=True, keep_input_mutations=aot_config.keep_inference_input_mutations, disable_amp=disable_amp)
    if not config.debug_assert:
        return compiled_function
    flat_requires_grad = [a.requires_grad if isinstance(a, Tensor) else None for a in flat_args]

    @wraps(compiled_function)
    def debug_compiled_function(*args):
        for i, a in enumerate(args):
            can_require_grad = flat_requires_grad[i]
            if can_require_grad is None:
                assert not isinstance(a, Tensor)
            elif not can_require_grad:
                assert not a.requires_grad, format_guard_bug_msg(aot_config, f'{describe_input(i, aot_config)} would not require grad')
        return compiled_function(*args)
    return debug_compiled_function
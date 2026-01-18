import contextlib
import dataclasses
import functools
import logging
import os
import sys
import time
import warnings
from itertools import count
from typing import (
from unittest import mock
from functorch.compile import min_cut_rematerialization_partition
import torch._functorch.config as functorch_config
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo import (
from torch._dynamo.utils import detect_fake_mode, lazy_format_graph_code
from torch._functorch.aot_autograd import aot_export_module, make_boxed_func
from torch._inductor.codecache import code_hash, CompiledFxGraph, FxGraphCache
from torch._inductor.debug import save_args_for_compile_fx_inner
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from .._dynamo.backends.common import aot_autograd
from ..fx.graph import _PyTreeCodeGen
from . import config, metrics
from .debug import DebugContext
from .decomposition import select_decomp_table
from .fx_passes.joint_graph import joint_graph_passes
from .fx_passes.post_grad import post_grad_passes, view_to_reshape
from .fx_passes.pre_grad import pre_grad_passes
from .graph import GraphLowering
from .ir import ExternKernelNode
from .utils import get_dtype_size, has_incompatible_cudagraph_ops
from .virtualized import V
@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
@time_and_log(attr='compilation time (in seconds)')
def compile_fx_inner(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], cudagraphs: Optional[BoxedBool]=None, num_fixed: int=0, is_backward: bool=False, graph_id: Optional[int]=None, cpp_wrapper: bool=False, aot_mode: bool=False, is_inference: bool=False, boxed_forward_device_index: Optional[BoxedDeviceIndex]=None, user_visible_outputs: FrozenSet[str]=frozenset(), layout_opt: Optional[bool]=None, extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]]=None) -> Union[CompiledFxGraph, str]:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """
    if dynamo_utils.count_calls(gm.graph) == 0 and (not aot_mode):
        return make_boxed_func(gm.forward)
    assert isinstance(next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)), f'inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}'
    if config.save_args:
        save_args_for_compile_fx_inner(gm, example_inputs, cudagraphs=cudagraphs, num_fixed=num_fixed, is_backward=is_backward, graph_id=graph_id, cpp_wrapper=cpp_wrapper, aot_mode=aot_mode, is_inference=is_inference, boxed_forward_device_index=boxed_forward_device_index, user_visible_outputs=user_visible_outputs, layout_opt=layout_opt)
    if cudagraphs is None:
        cudagraphs = BoxedBool(config.triton.cudagraphs)
    graph_kwargs = {'cudagraphs': cudagraphs, 'num_fixed': num_fixed, 'is_backward': is_backward, 'graph_id': graph_id, 'cpp_wrapper': cpp_wrapper, 'aot_mode': aot_mode, 'is_inference': is_inference, 'user_visible_outputs': user_visible_outputs, 'layout_opt': layout_opt, 'extern_node_serializer': extern_node_serializer}
    start = time.time()
    if config.fx_graph_cache and (not aot_mode):
        compiled_graph = FxGraphCache.load(fx_codegen_and_compile, gm, example_inputs, graph_kwargs)
    else:
        compiled_graph = fx_codegen_and_compile(gm, example_inputs, **graph_kwargs)
    log.debug('FX codegen and compilation took %.3fs', time.time() - start)
    context = torch._guards.TracingContext.try_get()
    if context is not None and context.output_strides is not None:
        assert len(context.output_strides) == 0
        context.output_strides.extend(compiled_graph.output_strides)
    if aot_mode:
        return compiled_graph
    if cudagraphs:
        output = list(gm.graph.nodes)[-1]
        assert len(output.args) == 1
        stack_traces = [arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None for arg in output.args[0]]
        complex_memory_overlap_inputs = any((complex_memory_overlap(t) for t in example_inputs if isinstance(t, torch.Tensor)))
        if config.triton.cudagraph_trees:
            has_mutation = not all((idx < num_fixed for idx in compiled_graph.mutated_input_idxs))
        else:
            has_mutation = len(compiled_graph.mutated_inputs) != 0
        cudagraph_tests = [(set(compiled_graph.device_types) == {'cuda'}, 'non-cuda device in graph'), (not has_mutation, 'mutated inputs'), (not has_incompatible_cudagraph_ops(gm), 'incompatible ops'), (not complex_memory_overlap_inputs, 'complex memory overlap'), (all((isinstance(t, (torch.Tensor, torch.SymInt)) for t in example_inputs)), 'non-Tensor inputs'), (len(compiled_graph.device_idxs) == 1 or not config.triton.cudagraph_trees, 'multiple device indices without cudagraph_trees')]
        cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]
        if not cudagraph_fail_reasons:
            if not config.triton.cudagraph_trees:
                for t in example_inputs:
                    if isinstance(t, torch.SymInt):
                        int(t)
            if boxed_forward_device_index is not None and (not is_inference) and (not is_backward):
                boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))
            compiled_graph.current_callable = cudagraphify(compiled_graph.get_current_callable(), example_inputs, static_input_idxs=range(num_fixed), device_index=next(iter(compiled_graph.device_idxs)), stack_traces=stack_traces, is_backward=is_backward, is_inference=is_inference, constants=tuple(compiled_graph.constants.values()))
        else:
            BoxedBool.disable(cudagraphs)
            if is_backward and config.triton.cudagraph_trees:
                assert boxed_forward_device_index is not None
                assert boxed_forward_device_index.value is not None
                compiled_graph_callable = compiled_graph.get_current_callable()
                manager = torch._inductor.cudagraph_trees.get_manager(boxed_forward_device_index.value, create_if_none_exists=False)
                assert manager is not None

                def compiled_artifact(new_inputs):
                    manager.set_to_running_backward()
                    return compiled_graph_callable(new_inputs)
                compiled_graph.current_callable = compiled_artifact
            if 'cuda' in compiled_graph.device_types:
                perf_hint_log.warning('skipping cudagraphs due to %s', cudagraph_fail_reasons)
    if not cudagraphs:
        new_callable = align_inputs(compiled_graph.get_current_callable(), example_inputs, range(num_fixed))
        if new_callable is not compiled_graph.get_current_callable():
            compiled_graph.current_callable = new_callable
    _step_logger()(logging.INFO, f'torchinductor done compiling {('BACKWARDS' if is_backward else 'FORWARDS')} graph {graph_id}')
    compiled_graph._boxed_call = True
    return compiled_graph
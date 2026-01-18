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
def fx_codegen_and_compile(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], cudagraphs: Optional[BoxedBool]=None, num_fixed: int=0, is_backward: bool=False, graph_id: Optional[int]=None, cpp_wrapper: bool=False, aot_mode: bool=False, is_inference: bool=False, user_visible_outputs: FrozenSet[str]=frozenset(), layout_opt: Optional[bool]=None, extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]]=None) -> Union[CompiledFxGraph, str]:
    if is_tf32_warning_applicable(gm):
        _warn_tf32_disabled()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
    _step_logger()(logging.INFO, f'torchinductor compiling {('BACKWARDS' if is_backward else 'FORWARDS')} graph {graph_id}')
    V.debug.fx_graph(gm, example_inputs)
    shape_env = _shape_env_from_inputs(example_inputs)
    view_to_reshape(gm)
    with torch.no_grad():
        fake_mode = fake_tensor_prop(gm, example_inputs)
    with V.set_fake_mode(fake_mode):
        post_grad_passes(gm, is_inference=is_inference)
        V.debug.fx_graph_transformed(gm, example_inputs)
        post_grad_graphs_log.info('%s', lazy_format_graph_code('AFTER POST GRAD', gm))
    with V.set_fake_mode(fake_mode):
        graph = GraphLowering(gm, example_inputs=V.real_inputs if is_inference else example_inputs, shape_env=shape_env, num_static_inputs=num_fixed, graph_id=graph_id, cpp_wrapper=cpp_wrapper, aot_mode=aot_mode, user_visible_outputs=user_visible_outputs, extern_node_serializer=extern_node_serializer, is_inference=is_inference)
        with V.set_graph_handler(graph):
            graph.run(*example_inputs)
            output_strides: List[Optional[Tuple[int, ...]]] = []
            if graph.graph_outputs is not None:
                for out in graph.graph_outputs:
                    if hasattr(out, 'layout'):
                        output_strides.append(tuple((V.graph.sizevars.size_hint(s) for s in out.layout.stride)))
                    else:
                        output_strides.append(None)
            compiled_fn = graph.compile_to_fn()
            if V.aot_compilation is True:
                return compiled_fn
            if graph.disable_cudagraphs:
                perf_hint_log.warning('skipping cudagraphs due to %s', V.graph.disable_cudagraphs_reason)
                BoxedBool.disable(cudagraphs)
            compiled_graph = CompiledFxGraph(compiled_fn, graph, output_strides)
    return compiled_graph
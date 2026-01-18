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
def compile_fx_aot(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor], inner_compile: Callable[..., Any]=compile_fx_inner, config_patches: Optional[Dict[str, Any]]=None):
    config_patches: Dict[str, Any] = {'cpp_wrapper': True} if config_patches is None else {**config_patches, 'cpp_wrapper': True}
    if 'aot_inductor.output_path' not in config_patches and (not config.aot_inductor.output_path):
        config_patches = {**config_patches, 'aot_inductor.output_path': code_hash(model_.code)}
    extern_node_serializer = config_patches.pop('extern_node_serializer', None)
    with V.set_aot_compilation(True):
        compiled_lib_path = compile_fx(model_, example_inputs_, inner_compile=functools.partial(inner_compile, aot_mode=True, extern_node_serializer=extern_node_serializer), config_patches=config_patches)
        assert os.path.exists(compiled_lib_path), f'AOTInductor compiled library does not exist at {compiled_lib_path}'
        return compiled_lib_path
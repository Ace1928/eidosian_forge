import dataclasses
import functools
from importlib import import_module
from typing import Any, List, Optional
from functorch.compile import min_cut_rematerialization_partition
import torch
from torch import _guards
from torch._functorch.compilers import ts_compile
from .common import aot_autograd
from .registry import register_debug_backend as register_backend
@register_backend
def non_leaf_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            break
    else:
        return gm
    for t in example_inputs:
        if not t.is_leaf:
            raise TestingOnlyCompileError()
    return gm
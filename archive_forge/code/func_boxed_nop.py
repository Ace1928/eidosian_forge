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
def boxed_nop(fx_g, example_inputs):

    def run(args):
        return torch.fx.Interpreter(fx_g).boxed_run(args)
    run._boxed_call = True
    return run
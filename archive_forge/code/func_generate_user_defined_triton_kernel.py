import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
def generate_user_defined_triton_kernel(self, kernel_name, grid, configs, args):
    assert len(grid) != 0
    if len(grid) == 1:
        grid_decision = grid[0]
    else:
        meta = CudaKernelParamCache.get(kernel_name)
        assert meta is not None
        grid_decision = None
        for i, c in enumerate(configs):
            if all((arg == meta['meta'][key] for key, arg in c.kwargs.items())):
                grid_decision = grid[i]
                break
        assert grid_decision is not None
    self.generate_kernel_call(kernel_name, args, grid=grid_decision, device_index=V.graph.scheduler.current_device.index, cuda=True, triton=True)
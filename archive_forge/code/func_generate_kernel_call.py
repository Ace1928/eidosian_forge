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
def generate_kernel_call(self, name, call_args, grid=None, device_index=None, cuda=True, triton=True):
    if not cuda:
        return super().generate_kernel_call(name, call_args, grid, device_index, cuda, triton)
    params = CudaKernelParamCache.get(name)
    assert params is not None, f'cuda kernel parameters for {name} should already exist at this moment'
    mangled_name = params.get('mangled_name', None)
    assert mangled_name is not None, 'missing mangled_name'
    cubin_path = params.get(get_cpp_wrapper_cubin_path_name(), None)
    assert cubin_path is not None and os.path.exists(cubin_path), f'cubin file should already exist at this moment: {cubin_path}'
    shared_mem = params.get('shared_mem', 0)
    self.generate_load_kernel_once(name, mangled_name, cubin_path, shared_mem)
    call_args = self.generate_args_decl(call_args)
    kernel_args_var = f'kernel_args_var_{next(self.kernel_callsite_id)}'
    self.writeline(f'void* {kernel_args_var}[] = {{{call_args}}};')
    stream = 'stream' if V.graph.aot_mode else self.write_get_raw_stream(device_index)
    grid_name = f'{name}_grid_{next(self.grid_id)}'
    assert isinstance(grid, (list, tuple)), f'expected grid to be a list or tuple but got: grid={grid!r}'
    grid = [V.graph.sizevars.simplify(item) for item in grid]
    grid_has_unbacked_symbols = any((free_unbacked_symbols(item) for item in grid))
    grid_args = [self.grid_expr_printer(item) for item in grid]
    grid_args_str = ', '.join(grid_args)
    self.writeline(f'Grid {grid_name} = Grid({grid_args_str});')
    if grid_has_unbacked_symbols:
        self.writeline(f'if ({grid_name}.is_non_zero()) {{')
    kernel_var_name = f'kernels.{name}' if V.graph.aot_mode else name
    self.writeline('launchKernel({}, {}, {}, {}, {}, {}, {}, {});'.format(kernel_var_name, f'{grid_name}.grid_x', f'{grid_name}.grid_y', f'{grid_name}.grid_z', params['num_warps'], params['shared_mem'], kernel_args_var, stream))
    if grid_has_unbacked_symbols:
        self.writeline('}')
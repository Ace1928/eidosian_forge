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
def benchmark_compiled_module(self, output):

    def add_fake_input(name, shape, stride, device, dtype):
        output.writeline(f"{name} = rand_strided({self.codegen_python_shape_tuple(shape)}, {self.codegen_python_shape_tuple(stride)}, device='{device}', dtype={dtype})")

    def add_expr_input(name, val):
        output.writeline(f'{name} = {val}')
    output.writelines(['', '', 'def benchmark_compiled_module(times=10, repeat=10):'])
    with output.indent():
        output.splice('\n                from torch._dynamo.testing import rand_strided\n                from torch._inductor.utils import print_performance\n                ', strip=True)
        for name, value in V.graph.constants.items():
            output.writeline(f'global {name}')
            add_fake_input(name, value.size(), value.stride(), value.device, value.dtype)
        for name, value in V.graph.graph_inputs.items():
            if isinstance(value, sympy.Symbol) and isinstance(V.graph.sizevars.var_to_val.get(value, None), SingletonInt):
                continue
            if isinstance(value, sympy.Expr):
                add_expr_input(name, V.graph.sizevars.size_hint(value))
            else:
                shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                add_fake_input(name, shape, stride, value.get_device(), value.get_dtype())
        call_str = f'call([{', '.join(V.graph.graph_inputs.keys())}])'
        output.writeline(f'fn = lambda: {call_str}')
        output.writeline('return print_performance(fn, times=times, repeat=repeat)')
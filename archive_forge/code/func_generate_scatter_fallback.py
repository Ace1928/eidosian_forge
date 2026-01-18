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
def generate_scatter_fallback(self, output, inputs, kernel, fn, src_is_tensor, reduce, kwargs):
    if V.graph.aot_mode and config.aot_inductor.abi_compatible:
        kernel = kernel.replace('at::', 'aoti_torch_')
    line = f'{kernel}({output}, {','.join(map(str, inputs))}'
    if fn == 'aten.scatter_':
        if src_is_tensor:
            if reduce:
                line += f', {V.graph.wrapper_code.val_to_arg_str(reduce)}'
        else:
            assert reduce is None, 'Expect reduce to be None for aten.scatter_ with scalar src'
    else:
        line += f', {','.join(kwargs)}'
    line += f'){self.ending}'
    self.writeline(line)
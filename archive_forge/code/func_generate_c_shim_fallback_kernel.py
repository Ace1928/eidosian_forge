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
def generate_c_shim_fallback_kernel(self, fallback_kernel, args):
    output_args = []
    output_raii_handles = []
    output_name_base = fallback_kernel.get_name()
    for idx, output in enumerate(fallback_kernel.outputs):
        if isinstance(output, ir.MultiOutput):
            name = f'{output.get_name()}'
            output_handle_name = f'{name}_handle'
            if output.indices:
                assert output.indices[0][1] == idx, f'expected output.indices[0][1]={output.indices[0][1]!r} == idx={idx!r} for output_name_base={output_name_base!r}'
            self.writeline(f'AtenTensorHandle {output_handle_name};')
            output_args.append(f'&{output_handle_name}')
            output_raii_handles.append(f'RAIIAtenTensorHandle {name}({output_handle_name});')
        elif isinstance(output, int):
            output_name = f'{output_name_base}_{idx}'
            self.writeline(f'int64_t {output_name} = {output};')
            output_args.append(f'&{output_name}')
        elif output is None:
            output_args.append('nullptr')
        else:
            raise NotImplementedError('unsupported type of {output=}')
    args = args + output_args
    assert fallback_kernel.abi_compatible_kernel is not None, f'abi_compatible_kernel is None for fallback_kernel.kernel={fallback_kernel.kernel!r}'
    self.generate_c_shim_extern_kernel_call(fallback_kernel.abi_compatible_kernel, args)
    for raii_handle in output_raii_handles:
        self.writeline(raii_handle)
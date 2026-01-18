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
def generate_c_shim_extern_kernel_alloc(self, extern_kernel, args):
    name = extern_kernel.name
    output_handle_name = f'{name}_handle'
    self.writeline(f'AtenTensorHandle {output_handle_name};')
    output_arg = f'&{output_handle_name}'
    self.generate_c_shim_extern_kernel_call(extern_kernel.codegen_kernel_name(), args + [output_arg])
    self.writeline(f'RAIIAtenTensorHandle {name}({output_handle_name});')
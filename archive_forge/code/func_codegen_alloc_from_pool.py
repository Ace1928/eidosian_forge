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
def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
    if config.aot_inductor.abi_compatible:
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(stride)
        tmp_name = f'tmp_tensor_handle_{next(self.tmp_tensor_id)}'
        args = [name, pexpr(offset), self.codegen_dtype(dtype), str(len(shape)), self.codegen_int_array_var(size, self.wrapper_call), self.codegen_int_array_var(stride, self.wrapper_call), f'&{tmp_name}']
        self.wrapper_call.writeline(f'AtenTensorHandle {tmp_name};')
        self.wrapper_call.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));')
        return f'RAIIAtenTensorHandle({tmp_name})'
    return 'alloc_from_pool({})'.format(', '.join([name, pexpr(offset), self.codegen_dtype(dtype), self.codegen_shape_tuple(shape), self.codegen_shape_tuple(stride)]))
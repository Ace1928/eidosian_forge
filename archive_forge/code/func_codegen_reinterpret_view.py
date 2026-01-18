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
def codegen_reinterpret_view(self, data, size_list, stride_list, offset, writer) -> str:
    dim = str(len(size_list))
    size = self.codegen_shape_tuple(size_list)
    stride = self.codegen_shape_tuple(stride_list)
    offset = self.codegen_sizevar(offset)
    if config.aot_inductor.abi_compatible:
        tmp_name = f'tmp_tensor_handle_{next(self.tmp_tensor_id)}'
        if writer is None:
            writer = self
        args = [f'{data.get_name()}', dim, self.codegen_int_array_var(size, writer), self.codegen_int_array_var(stride, writer), offset, f'&{tmp_name}']

        def gen_reinterpret_call(writer, args):
            writer.writeline(f'AtenTensorHandle {tmp_name};')
            writer.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor({', '.join(args)}));')
        if self.can_cache_buffer_in_thread_local(data) and self.is_statically_known_list_of_ints(size_list) and self.is_statically_known_list_of_ints(stride_list):
            self.cached_thread_locals.add(tmp_name)
            writer.writeline(f'thread_local RAIIAtenTensorHandle {tmp_name}_handle = ([&] {{')
            if hasattr(writer, 'indent'):
                indent = writer.indent()
            else:
                indent = contextlib.nullcontext()
            with indent:
                gen_reinterpret_call(writer, args)
                writer.writeline(f'return {tmp_name};')
            writer.writeline('})();')
            writer.writeline(f'AtenTensorHandle {tmp_name}({tmp_name}_handle.get());')
            return tmp_name
        gen_reinterpret_call(writer, args)
        return f'RAIIAtenTensorHandle({tmp_name})'
    else:
        args = [data.get_name(), size, stride, offset]
        return f'reinterpret_tensor({', '.join(args)})'
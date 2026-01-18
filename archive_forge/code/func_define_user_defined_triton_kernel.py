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
def define_user_defined_triton_kernel(self, kernel, configs, kwargs):
    original_name = kernel.__name__
    cache_key = [id(kernel.fn)]
    for arg in kwargs.values():
        if isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
            cache_key.append(arg.get_dtype())
        elif len(configs) > 0:
            cache_key.append(arg)
    cache_key = tuple(cache_key)
    if cache_key in self.user_defined_kernel_cache:
        return self.user_defined_kernel_cache[cache_key]
    name = f'{original_name}_{len(self.user_defined_kernel_cache)}'
    self.user_defined_kernel_cache[cache_key] = name
    compile_wrapper = IndentedBuffer()
    compile_wrapper.writeline(f"async_compile.triton({original_name!r}, '''")
    compile_wrapper.splice('\n            import triton\n            import triton.language as tl\n            from torch._inductor.utils import instance_descriptor\n            from torch._inductor.triton_heuristics import user_autotune\n            ', strip=True)
    compile_wrapper.newline()
    from .common import SizeArg, TensorArg
    signature: List[Union[TensorArg, SizeArg]] = []
    constants = {}
    for key, arg in kwargs.items():
        idx = kernel.arg_names.index(key)
        if idx in kernel.constexprs:
            constants[key] = arg
            continue
        if isinstance(arg, (ir.Buffer, ir.ReinterpretView)):
            signature.append(TensorArg(key, arg.codegen_reference(), arg.get_dtype(), not isinstance(arg, ReinterpretView)))
        else:
            signature.append(SizeArg(key, arg))
    index_dtype = 'tl.int32'
    inductor_meta = {'kernel_name': name}
    triton_meta = {'signature': signature_to_meta(signature, size_dtype=index_dtype), 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': constants, 'configs': [config_of(signature)]}
    configs = [{'kwargs': config.kwargs, 'num_warps': config.num_warps, 'num_stages': config.num_stages} for config in configs]
    compile_wrapper.splice(f'\n            @user_autotune(\n                configs={configs!r},\n                inductor_meta={inductor_meta!r},\n                triton_meta={triton_meta!r},\n                filename=__file__\n            )\n            @triton.jit\n            ')
    compile_wrapper.splice(kernel.src, strip=True)
    from triton import JITFunction
    symbols_included = {original_name}

    def traverse(cur_kernel):
        for symbol_name in cur_kernel.fn.__code__.co_names:
            if symbol_name in symbols_included:
                continue
            if symbol_name in cur_kernel.fn.__globals__:
                symbol = cur_kernel.fn.__globals__[symbol_name]
                if isinstance(symbol, JITFunction):
                    compile_wrapper.newline()
                    compile_wrapper.writeline('@triton.jit')
                    compile_wrapper.splice(symbol.src, strip=True)
                    symbols_included.add(symbol_name)
                    traverse(symbol)
                elif isinstance(symbol, (int, str, bool)):
                    compile_wrapper.newline()
                    compile_wrapper.writeline(f'{symbol_name} = {symbol!r}')
                    symbols_included.add(symbol_name)
    traverse(kernel)
    compile_wrapper.writeline("''')")
    _, lineno = inspect.getsourcelines(kernel.fn)
    srcfile = inspect.getsourcefile(kernel.fn)
    metadata = f'# Original path: {srcfile}:{lineno}'
    self.define_kernel(name, compile_wrapper.getvalue(), metadata)
    return name
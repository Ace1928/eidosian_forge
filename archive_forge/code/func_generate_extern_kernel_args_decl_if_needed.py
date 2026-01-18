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
def generate_extern_kernel_args_decl_if_needed(self, op_overload, raw_args, output_args):
    arg_types = [x.real_type for x in op_overload._schema.arguments]
    return_types = [x.type for x in op_overload._schema.returns]
    new_tensor_args = []
    new_int_args = []

    def fill_args(arg, arg_type):
        static_arg_types = (torch.FloatType, torch.BoolType, torch.StringType, torch.Type, torch.DeviceObjType)
        inductor_tensor_buffers = (ir.Buffer, ir.ReinterpretView)
        if isinstance(arg_type, torch.TensorType):
            assert isinstance(arg, inductor_tensor_buffers), f'got {type(arg)}'
            new_tensor_args.append(f'{arg.codegen_reference()}')
        elif isinstance(arg_type, torch.IntType):
            new_int_args.append(str(arg))
        elif isinstance(arg_type, torch.SymIntType):
            new_int_args.append(str(arg))
        elif isinstance(arg_type, torch.NumberType):
            assert isinstance(arg, (int, float, bool))
            if isinstance(arg, int):
                new_int_args.append(str(arg))
        elif isinstance(arg_type, torch.ListType):
            assert isinstance(arg, (list, tuple))
            if isinstance(arg_type.getElementType(), torch.TensorType):
                new_tensor_args.extend([f'{a.codegen_reference()}' for a in arg])
            elif isinstance(arg_type.getElementType(), torch.OptionalType) and isinstance(arg_type.getElementType().getElementType(), torch.TensorType):
                new_tensor_args.extend([f'{a.codegen_reference()}' for a in arg if a is not None])
            elif isinstance(arg_type.getElementType(), (torch.IntType, torch.SymIntType)):
                new_int_args.extend([str(a) for a in arg])
            elif isinstance(arg_type.getElementType(), torch.NumberType):
                is_int_type = [isinstance(a, int) for a in arg]
                if any(is_int_type):
                    assert all(is_int_type), 'AOTInductor only supports int scalars of the same type'
                    new_int_args.extend([str(a) for a in arg])
            else:
                assert isinstance(arg_type.getElementType(), static_arg_types), f'Fall through arguments must be one of static_arg_types, got {type(arg_type)}'
        else:
            assert isinstance(arg_type, static_arg_types), f'Fall through arguments must be one of static_arg_types, got {type(arg_type)}'
    for arg, arg_type in zip(raw_args, arg_types):
        if arg is not None:
            if isinstance(arg_type, torch.OptionalType):
                fill_args(arg, arg_type.getElementType())
            else:
                fill_args(arg, arg_type)

    def fill_output_arg(arg, return_type):
        if isinstance(return_type, torch.TensorType):
            self.writeline(f'AtenTensorHandle {arg}_handle;  // output buffer')
            self.writeline(f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&{arg}_handle));')
            self.writeline(f'RAIIAtenTensorHandle {arg}({arg}_handle);')
            new_tensor_args.append(f'{arg}')
        elif isinstance(return_type, torch.SymIntType):
            raise NotImplementedError('NYI support for return type: SymInt')
        elif isinstance(return_type, torch.ListType) and isinstance(return_type.getElementType(), torch.SymIntType):
            raise NotImplementedError('NYI support for return type: List[SymInt]')
        else:
            raise AssertionError(f'Unsupported return type found: {return_type}')
    for return_type in return_types:
        if isinstance(return_type, torch.TensorType):
            pass
        elif isinstance(return_type, torch.OptionalType):
            assert isinstance(return_type.getElementType(), torch.TensorType)
        elif isinstance(return_type, torch.ListType):
            assert isinstance(return_type.getElementType(), torch.TensorType)
        else:
            raise NotImplementedError(f'return type {return_type} is not yet supported.')
    for output_arg in output_args:
        assert output_arg is not None, 'Optional return types are not yet supported'
        if isinstance(output_arg, (list, tuple)):
            for out in output_arg:
                fill_output_arg(out, torch.TensorType.get())
        else:
            fill_output_arg(output_arg, torch.TensorType.get())
    return (new_tensor_args, new_int_args)
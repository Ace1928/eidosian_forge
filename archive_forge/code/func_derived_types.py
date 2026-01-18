import dataclasses
import functools
import inspect
import sys
import typing
import weakref
from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy
import torch
import torch._C as _C
import torch.library as library
from torch._library.abstract_impl import AbstractImplCtx
from torch.library import get_ctx
from .autograd import autograd_kernel_indirection, construct_autograd_kernel
def derived_types(base_type, cpp_type, list_base, optional_base_list, optional_list_base):
    result = [(base_type, cpp_type), (typing.Optional[base_type], f'{cpp_type}?')]
    if list_base:
        result.append((typing.Sequence[base_type], f'{cpp_type}[]'))
    if optional_base_list:
        result.append((typing.Sequence[typing.Optional[base_type]], f'{cpp_type}?[]'))
    if optional_list_base:
        result.append((typing.Optional[typing.Sequence[base_type]], f'{cpp_type}[]?'))
    return result
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
def parse_return(annotation, error_fn):
    origin = typing.get_origin(annotation)
    if origin is not tuple:
        if annotation not in SUPPORTED_RETURN_TYPES.keys():
            error_fn(f'Return has unsupported type {annotation}. The valid types are: {SUPPORTED_RETURN_TYPES}.')
        return SUPPORTED_RETURN_TYPES[annotation]
    args = typing.get_args(annotation)
    for arg in args:
        if arg not in SUPPORTED_RETURN_TYPES:
            error_fn(f'Return has unsupported type {annotation}. The valid types are: {SUPPORTED_RETURN_TYPES}.')
    return '(' + ', '.join([SUPPORTED_RETURN_TYPES[arg] for arg in args]) + ')'
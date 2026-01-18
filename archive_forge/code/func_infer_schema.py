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
def infer_schema(prototype_function: typing.Callable) -> str:
    sig = inspect.signature(prototype_function)

    def error_fn(what):
        raise ValueError(f'custom_op(...)(func): {what} Got func with signature {sig})')
    params = [parse_param(name, param, error_fn) for name, param in sig.parameters.items()]
    ret = parse_return(sig.return_annotation, error_fn)
    return f'({', '.join(params)}) -> {ret}'
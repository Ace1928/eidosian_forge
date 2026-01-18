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
def parse_qualname(qualname: str) -> typing.Tuple[str, str]:
    names = qualname.split('::', 1)
    if len(names) != 2:
        raise ValueError(f'Expected there to be a namespace in {qualname}, i.e. The operator name should look something like ns::foo')
    if '.' in names[1]:
        raise ValueError(f"The torch.custom_ops APIs do not handle overloads, i.e. operator names with '.' in them. Please name your operator something like ns::foo. Got: {qualname}")
    return (names[0], names[1])
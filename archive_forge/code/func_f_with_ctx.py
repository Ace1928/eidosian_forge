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
@functools.wraps(f)
def f_with_ctx(*args, **kwargs):

    def error_on_ctx():
        raise RuntimeError(f'Attempted to call get_ctx() for the meta implementation for {qualname}.You have presumably called get_ctx() because the operator has a data-dependent output shape; if so, there is no such meta implementation and this error is the correct behavior. Otherwise, please remove the call to get_ctx() in the implementation registered with impl_abstract at {location}')
    with torch._library.abstract_impl.set_ctx_getter(error_on_ctx):
        return f(*args, **kwargs)
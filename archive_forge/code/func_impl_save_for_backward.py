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
def impl_save_for_backward(self, _stacklevel=2):
    """Register a function that tells us what to save for backward.

        Please see impl_backward for more details.
        """

    def inner(f):
        self._check_can_register_backward()
        self._check_doesnt_have_library_autograd_impl()
        if not self._registered_autograd_kernel_indirection:
            self._register_autograd_kernel_indirection()
        self._register_impl('save_for_backward', f, stacklevel=_stacklevel)
        if self._has_impl('backward'):
            self._register_autograd_kernel()
    return inner
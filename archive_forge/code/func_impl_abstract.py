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
def impl_abstract(self, _stacklevel=2) -> typing.Callable:
    """Register an abstract implementation for this operator.

        WARNING: please do not use this directly (and instead use the torch._custom_ops
        APIs). Also please see the following for a detailed guide on custom ops.
        https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

        An "abstract implementation" specifies the behavior of this operator on
        Tensors that carry no data. Given some input Tensors with certain properties
        (sizes/strides/storage_offset/device), it specifies what the properties of
        the output Tensors are.

        The abstract implementation has the same signature as the operator.
        It is run for both FakeTensors and meta tensors. To write an abstract
        implementation, assume that all Tensor inputs to the operator are
        regular CPU/CUDA/Meta tensors, but they do not have storage, and
        you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
        The abstract implementation must consist of only PyTorch operations
        (and may not directly access the storage or data of any input or
        intermediate Tensors).

        This API is used as a decorator (see examples).

        Examples::
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> # Example 1: an operator without data-dependent output shape
            >>> @custom_op('my_library::custom_linear')
            >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            >>>     ...
            >>>
            >>> @custom_linear.impl_abstract()
            >>> def custom_linear_abstract(x, weight):
            >>>     assert x.dim() == 2
            >>>     assert weight.dim() == 2
            >>>     assert bias.dim() == 1
            >>>     assert x.shape[1] == weight.shape[1]
            >>>     assert weight.shape[0] == bias.shape[0]
            >>>     assert x.device == weight.device
            >>>
            >>>     return (x @ weight.t()) + bias
            >>>
            >>> # Example 2: an operator with data-dependent output shape
            >>> @custom_op('my_library::custom_nonzero')
            >>> def custom_nonzero(x: Tensor) -> Tensor:
            >>>     ...
            >>>
            >>> @custom_nonzero.impl_abstract()
            >>> def custom_nonzero_abstract(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in an abstract impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch._custom_op.get_ctx()
            >>>     nnz = ctx.create_unbacked_symint()
            >>>     shape = [x.dim(), nnz]
            >>>     result = x.new_empty(shape, dtype=torch.long)
            >>>     return result
            >>>
            >>> @custom_nonzero.impl(['cpu', 'cuda'])
            >>> def custom_nonzero_impl(x):
            >>>     x_np = to_numpy(x)
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     # unbacked symbolic ints in PyTorch must be >= 2, so we
            >>>     # constrain the range to at least 2
            >>>     if res.shape[0] <= 1:
            >>>         raise RuntimeError("not supported")
            >>>     return torch.tensor(res, device=x.device)

        """

    def inner(f):
        self._check_doesnt_have_library_meta_impl()
        self._register_impl('abstract', f, stacklevel=_stacklevel)
        location = self._get_impl('abstract').location
        qualname = self._qualname

        @functools.wraps(f)
        def f_with_ctx(*args, **kwargs):

            def error_on_ctx():
                raise RuntimeError(f'Attempted to call get_ctx() for the meta implementation for {qualname}.You have presumably called get_ctx() because the operator has a data-dependent output shape; if so, there is no such meta implementation and this error is the correct behavior. Otherwise, please remove the call to get_ctx() in the implementation registered with impl_abstract at {location}')
            with torch._library.abstract_impl.set_ctx_getter(error_on_ctx):
                return f(*args, **kwargs)
        self._lib.impl(self._opname, f_with_ctx, 'Meta')
        return f
    return inner
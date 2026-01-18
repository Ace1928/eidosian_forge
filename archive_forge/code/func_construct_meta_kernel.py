import contextlib
import functools
import warnings
from typing import Callable, Optional
import torch
from torch._library.utils import Kernel, RegistrationHandle
def construct_meta_kernel(qualname: str, abstract_impl_holder: AbstractImplHolder) -> Callable:
    assert abstract_impl_holder.kernel is not None

    @functools.wraps(abstract_impl_holder.kernel.func)
    def meta_kernel(*args, **kwargs):
        assert abstract_impl_holder.kernel is not None
        source = abstract_impl_holder.kernel.source

        def error_on_ctx():
            raise RuntimeError(f'Attempted to call get_ctx() for the meta implementation for {qualname} (implemented at {source})You have presumably called get_ctx() because the operator has a data-dependent output shape; if so, there is no such meta implementation and this error is the correct behavior.')
        with set_ctx_getter(error_on_ctx):
            return abstract_impl_holder.kernel(*args, **kwargs)
    return meta_kernel
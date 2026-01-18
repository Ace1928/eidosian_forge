from functools import partial
import torch
from .binary import (
from .core import is_masked_tensor, MaskedTensor, _get_data, _masks_match, _maybe_get_mask
from .passthrough import (
from .reductions import (
from .unary import (
def register_function_func(ops):
    """
    Used for registering a new __torch_function__ function to MaskedTensor
    Called via _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

    The code to register a new function looks like:

    @register_function_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(func):
        for op in ops:
            _MASKEDTENSOR_FUNCTION_TABLE[op] = partial(func, op)
    return wrapper
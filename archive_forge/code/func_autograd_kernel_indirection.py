import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
def autograd_kernel_indirection(custom_op):
    autograd_fallback = autograd_not_implemented(custom_op)

    def inner(*args, **kwargs):
        if custom_op._has_impl('autograd'):
            kernel = custom_op._get_impl('autograd').func
            return kernel(*args, **kwargs)
        if custom_op._has_impl('save_for_backward') or custom_op._has_impl('backward'):
            missing = 'save_for_backward' if custom_op._has_impl('backward') else 'backward'
            found = 'save_for_backward' if missing == 'backward' else 'backward'
            loc = custom_op._get_impl(found).location
            raise RuntimeError(f"We found a '{found}' registration for {custom_op} at {loc} but were unable to find a '{missing}' registration. To use the CustomOp API to register a backward formula, please provide us both a backward function and a 'save for backward' function via `impl_backward` and `impl_save_for_backward` respectively.")
        return autograd_fallback(*args, **kwargs)
    return inner
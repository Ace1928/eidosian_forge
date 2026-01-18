from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
def guard_invalid_slice(context, builder, typ, slicestruct):
    """
    Guard against *slicestruct* having a zero step (and raise ValueError).
    """
    if typ.has_step:
        cgutils.guard_null(context, builder, slicestruct.step, (ValueError, 'slice step cannot be zero'))
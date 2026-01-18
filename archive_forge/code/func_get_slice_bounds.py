from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
def get_slice_bounds(builder, slicestruct):
    """
    Return the [lower, upper) indexing bounds of a slice.
    """
    start = slicestruct.start
    stop = slicestruct.stop
    zero = start.type(0)
    one = start.type(1)
    is_step_negative = builder.icmp_signed('<', slicestruct.step, zero)
    lower = builder.select(is_step_negative, builder.add(stop, one), start)
    upper = builder.select(is_step_negative, builder.add(start, one), stop)
    return (lower, upper)
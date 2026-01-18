from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
def fix_bound(bound_name, lower_repl, upper_repl):
    bound = getattr(slice, bound_name)
    bound = fix_index(builder, bound, size)
    setattr(slice, bound_name, bound)
    underflow = builder.icmp_signed('<', bound, zero)
    with builder.if_then(underflow, likely=False):
        setattr(slice, bound_name, lower_repl)
    overflow = builder.icmp_signed('>=', bound, size)
    with builder.if_then(overflow, likely=False):
        setattr(slice, bound_name, upper_repl)
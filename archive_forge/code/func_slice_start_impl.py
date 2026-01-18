from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
@lower_getattr(types.SliceType, 'start')
def slice_start_impl(context, builder, typ, value):
    sli = context.make_helper(builder, typ, value)
    return sli.start
from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
@lower_getattr(types.SliceType, 'step')
def slice_step_impl(context, builder, typ, value):
    if typ.has_step:
        sli = context.make_helper(builder, typ, value)
        return sli.step
    else:
        return context.get_constant(types.intp, 1)
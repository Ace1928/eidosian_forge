from functools import singledispatch
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.errors import NumbaWarning
from numba.core.imputils import Registry
from numba.cuda import nvvmutils
from warnings import warn
@print_item.register(types.Float)
def real_print_impl(ty, context, builder, val):
    lld = context.cast(builder, val, ty, types.float64)
    return ('%f', [lld])
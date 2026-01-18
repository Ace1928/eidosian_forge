from functools import singledispatch
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.errors import NumbaWarning
from numba.core.imputils import Registry
from numba.cuda import nvvmutils
from warnings import warn
@print_item.register(types.Integer)
@print_item.register(types.IntegerLiteral)
def int_print_impl(ty, context, builder, val):
    if ty in types.unsigned_domain:
        rawfmt = '%llu'
        dsttype = types.uint64
    else:
        rawfmt = '%lld'
        dsttype = types.int64
    lld = context.cast(builder, val, ty, dsttype)
    return (rawfmt, [lld])
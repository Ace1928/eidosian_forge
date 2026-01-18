from functools import singledispatch
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.errors import NumbaWarning
from numba.core.imputils import Registry
from numba.cuda import nvvmutils
from warnings import warn
@print_item.register(types.StringLiteral)
def const_print_impl(ty, context, builder, sigval):
    pyval = ty.literal_value
    assert isinstance(pyval, str)
    rawfmt = '%s'
    val = context.insert_string_const_addrspace(builder, pyval)
    return (rawfmt, [val])
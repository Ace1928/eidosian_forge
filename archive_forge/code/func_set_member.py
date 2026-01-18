from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
def set_member(member_offset, value):
    offset = c.context.get_constant(types.uintp, member_offset)
    ptr = cgutils.pointer_add(c.builder, box, offset)
    casted = c.builder.bitcast(ptr, llvoidptr.as_pointer())
    c.builder.store(value, casted)
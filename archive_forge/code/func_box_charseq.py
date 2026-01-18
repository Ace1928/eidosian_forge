from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.CharSeq)
def box_charseq(typ, val, c):
    rawptr = cgutils.alloca_once_value(c.builder, value=val)
    strptr = c.builder.bitcast(rawptr, c.pyapi.cstring)
    fullsize = c.context.get_constant(types.intp, typ.count)
    zero = fullsize.type(0)
    one = fullsize.type(1)
    count = cgutils.alloca_once_value(c.builder, zero)
    with cgutils.loop_nest(c.builder, [fullsize], fullsize.type) as [idx]:
        ch = c.builder.load(c.builder.gep(strptr, [idx]))
        with c.builder.if_then(cgutils.is_not_null(c.builder, ch)):
            c.builder.store(c.builder.add(idx, one), count)
    strlen = c.builder.load(count)
    return c.pyapi.bytes_from_string_and_size(strptr, strlen)
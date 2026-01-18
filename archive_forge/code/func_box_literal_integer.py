from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.IntegerLiteral)
@box(types.BooleanLiteral)
def box_literal_integer(typ, val, c):
    val = c.context.cast(c.builder, val, typ, typ.literal_type)
    return c.box(typ.literal_type, val)
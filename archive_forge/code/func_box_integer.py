from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.Integer)
def box_integer(typ, val, c):
    if typ.signed:
        ival = c.builder.sext(val, c.pyapi.longlong)
        return c.pyapi.long_from_longlong(ival)
    else:
        ullval = c.builder.zext(val, c.pyapi.ulonglong)
        return c.pyapi.long_from_ulonglong(ullval)
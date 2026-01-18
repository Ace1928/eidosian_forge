from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.Float)
def box_float(typ, val, c):
    if typ == types.float32:
        dbval = c.builder.fpext(val, c.pyapi.double)
    else:
        assert typ == types.float64
        dbval = val
    return c.pyapi.float_from_double(dbval)
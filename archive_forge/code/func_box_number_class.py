from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.NumberClass)
def box_number_class(typ, val, c):
    np_dtype = numpy_support.as_dtype(typ.dtype)
    return c.pyapi.unserialize(c.pyapi.serialize_object(np_dtype))
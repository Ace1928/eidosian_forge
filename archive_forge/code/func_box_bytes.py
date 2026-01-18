from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.Bytes)
def box_bytes(typ, val, c):
    obj = c.context.make_helper(c.builder, typ, val)
    ret = c.pyapi.bytes_from_string_and_size(obj.data, obj.nitems)
    c.context.nrt.decref(c.builder, typ, val)
    return ret
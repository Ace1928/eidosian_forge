from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.RawPointer)
def box_raw_pointer(typ, val, c):
    """
    Convert a raw pointer to a Python int.
    """
    ll_intp = c.context.get_value_type(types.uintp)
    addr = c.builder.ptrtoint(val, ll_intp)
    return c.box(types.uintp, addr)
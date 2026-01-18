from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.Tuple)
@box(types.UniTuple)
def box_tuple(typ, val, c):
    """
    Convert native array or structure *val* to a tuple object.
    """
    tuple_val = c.pyapi.tuple_new(typ.count)
    for i, dtype in enumerate(typ):
        item = c.builder.extract_value(val, i)
        obj = c.box(dtype, item)
        c.pyapi.tuple_setitem(tuple_val, i, obj)
    return tuple_val
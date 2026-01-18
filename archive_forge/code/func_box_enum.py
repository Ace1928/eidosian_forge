from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.EnumMember)
def box_enum(typ, val, c):
    """
    Fetch an enum member given its native value.
    """
    valobj = c.box(typ.dtype, val)
    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.instance_class))
    return c.pyapi.call_function_objargs(cls_obj, (valobj,))
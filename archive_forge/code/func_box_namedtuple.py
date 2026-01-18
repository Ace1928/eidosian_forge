from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.NamedTuple)
@box(types.NamedUniTuple)
def box_namedtuple(typ, val, c):
    """
    Convert native array or structure *val* to a namedtuple object.
    """
    cls_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.instance_class))
    tuple_obj = box_tuple(typ, val, c)
    obj = c.pyapi.call(cls_obj, tuple_obj)
    c.pyapi.decref(cls_obj)
    c.pyapi.decref(tuple_obj)
    return obj
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@reflect(types.Set)
def reflect_set(typ, val, c):
    """
    Reflect the native set's contents into the Python object.
    """
    if not typ.reflected:
        return
    inst = setobj.SetInstance(c.context, c.builder, typ, val)
    payload = inst.payload
    with c.builder.if_then(payload.dirty, likely=False):
        obj = inst.parent
        c.pyapi.set_clear(obj)
        ok, listobj = _native_set_to_python_list(typ, payload, c)
        with c.builder.if_then(ok, likely=True):
            c.pyapi.set_update(obj, listobj)
            c.pyapi.decref(listobj)
        inst.set_dirty(False)
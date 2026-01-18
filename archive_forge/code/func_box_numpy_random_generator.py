from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.NumPyRandomGeneratorType)
def box_numpy_random_generator(typ, val, c):
    inst = c.context.make_helper(c.builder, typ, val)
    obj = inst.parent
    res = cgutils.alloca_once_value(c.builder, obj)
    c.pyapi.incref(obj)
    c.context.nrt.decref(c.builder, typ, val)
    return c.builder.load(res)
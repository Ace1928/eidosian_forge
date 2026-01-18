from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
def object_getattr_safely(obj, attr):
    attr_obj = c.pyapi.object_getattr_string(obj, attr)
    extra_refs.append(attr_obj)
    return attr_obj
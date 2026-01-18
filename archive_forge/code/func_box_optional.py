from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
@box(types.Optional)
def box_optional(typ, val, c):
    optval = c.context.make_helper(c.builder, typ, val)
    ret = cgutils.alloca_once_value(c.builder, c.pyapi.borrow_none())
    with c.builder.if_else(optval.valid) as (then, otherwise):
        with then:
            validres = c.box(typ.type, optval.data)
            c.builder.store(validres, ret)
        with otherwise:
            c.builder.store(c.pyapi.make_none(), ret)
    return c.builder.load(ret)
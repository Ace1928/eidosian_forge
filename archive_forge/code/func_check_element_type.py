from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
def check_element_type(nth, itemobj, expected_typobj):
    typobj = nth.typeof(itemobj)
    with c.builder.if_then(cgutils.is_null(c.builder, typobj), likely=False):
        c.builder.store(cgutils.true_bit, errorptr)
        loop.do_break()
    type_mismatch = c.builder.icmp_signed('!=', typobj, expected_typobj)
    with c.builder.if_then(type_mismatch, likely=False):
        c.builder.store(cgutils.true_bit, errorptr)
        c.pyapi.err_format('PyExc_TypeError', "can't unbox heterogeneous list: %S != %S", expected_typobj, typobj)
        c.pyapi.decref(typobj)
        loop.do_break()
    c.pyapi.decref(typobj)
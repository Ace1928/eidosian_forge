import sys
import operator
import numpy as np
from llvmlite.ir import IntType, Constant
from numba.core.cgutils import is_nonelike
from numba.core.extending import (
from numba.core.imputils import (lower_constant, lower_cast, lower_builtin,
from numba.core.datamodel import register_default, StructModel
from numba.core import types, cgutils
from numba.core.utils import PYVERSION
from numba.core.pythonapi import (
from numba._helperlib import c_helpers
from numba.cpython.hashing import _Py_hash_t
from numba.core.unsafe.bytes import memcpy_region
from numba.core.errors import TypingError
from numba.cpython.unicode_support import (_Py_TOUPPER, _Py_TOLOWER, _Py_UCS4,
from numba.cpython import slicing
@lower_builtin('getiter', types.UnicodeType)
def getiter_unicode(context, builder, sig, args):
    [ty] = sig.args
    [data] = args
    iterobj = context.make_helper(builder, sig.return_type)
    zero = context.get_constant(types.uintp, 0)
    indexptr = cgutils.alloca_once_value(builder, zero)
    iterobj.index = indexptr
    iterobj.data = data
    if context.enable_nrt:
        context.nrt.incref(builder, ty, data)
    res = iterobj._getvalue()
    return impl_ret_new_ref(context, builder, sig.return_type, res)
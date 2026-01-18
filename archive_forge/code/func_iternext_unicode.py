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
@lower_builtin('iternext', types.UnicodeIteratorType)
@iternext_impl(RefType.NEW)
def iternext_unicode(context, builder, sig, args, result):
    [iterty] = sig.args
    [iter] = args
    tyctx = context.typing_context
    fnty = tyctx.resolve_value_type(operator.getitem)
    getitem_sig = fnty.get_call_type(tyctx, (types.unicode_type, types.uintp), {})
    getitem_impl = context.get_function(fnty, getitem_sig)
    fnty = tyctx.resolve_value_type(len)
    len_sig = fnty.get_call_type(tyctx, (types.unicode_type,), {})
    len_impl = context.get_function(fnty, len_sig)
    iterobj = context.make_helper(builder, iterty, value=iter)
    strlen = len_impl(builder, (iterobj.data,))
    index = builder.load(iterobj.index)
    is_valid = builder.icmp_unsigned('<', index, strlen)
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        gotitem = getitem_impl(builder, (iterobj.data, index))
        result.yield_(gotitem)
        nindex = cgutils.increment_index(builder, index)
        builder.store(nindex, iterobj.index)
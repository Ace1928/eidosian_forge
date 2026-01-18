import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def call_xxdot(context, builder, conjugate, dtype, n, a_data, b_data, out_data):
    """
    Call the BLAS vector * vector product function for the given arguments.
    """
    fnty = ir.FunctionType(ir.IntType(32), [ll_char, ll_char, intp_t, ll_void_p, ll_void_p, ll_void_p])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_xxdot')
    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))
    conjugate = ir.Constant(ll_char, int(conjugate))
    res = builder.call(fn, (kind_val, conjugate, n, builder.bitcast(a_data, ll_void_p), builder.bitcast(b_data, ll_void_p), builder.bitcast(out_data, ll_void_p)))
    check_blas_return(context, builder, res)
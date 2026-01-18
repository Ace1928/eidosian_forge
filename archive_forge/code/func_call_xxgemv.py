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
def call_xxgemv(context, builder, do_trans, m_type, m_shapes, m_data, v_data, out_data):
    """
    Call the BLAS matrix * vector product function for the given arguments.
    """
    fnty = ir.FunctionType(ir.IntType(32), [ll_char, ll_char, intp_t, intp_t, ll_void_p, ll_void_p, intp_t, ll_void_p, ll_void_p, ll_void_p])
    fn = cgutils.get_or_insert_function(builder.module, fnty, 'numba_xxgemv')
    dtype = m_type.dtype
    alpha = make_constant_slot(context, builder, dtype, 1.0)
    beta = make_constant_slot(context, builder, dtype, 0.0)
    if m_type.layout == 'F':
        m, n = m_shapes
        lda = m_shapes[0]
    else:
        n, m = m_shapes
        lda = m_shapes[1]
    kind = get_blas_kind(dtype)
    kind_val = ir.Constant(ll_char, ord(kind))
    trans = ir.Constant(ll_char, ord('t') if do_trans else ord('n'))
    res = builder.call(fn, (kind_val, trans, m, n, builder.bitcast(alpha, ll_void_p), builder.bitcast(m_data, ll_void_p), lda, builder.bitcast(v_data, ll_void_p), builder.bitcast(beta, ll_void_p), builder.bitcast(out_data, ll_void_p)))
    check_blas_return(context, builder, res)
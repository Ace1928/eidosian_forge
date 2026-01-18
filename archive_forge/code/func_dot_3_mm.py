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
def dot_3_mm(context, builder, sig, args):
    """
    np.dot(matrix, matrix, out)
    """
    xty, yty, outty = sig.args
    assert outty == sig.return_type
    dtype = xty.dtype
    x = make_array(xty)(context, builder, args[0])
    y = make_array(yty)(context, builder, args[1])
    out = make_array(outty)(context, builder, args[2])
    x_shapes = cgutils.unpack_tuple(builder, x.shape)
    y_shapes = cgutils.unpack_tuple(builder, y.shape)
    out_shapes = cgutils.unpack_tuple(builder, out.shape)
    m, k = x_shapes
    _k, n = y_shapes
    assert outty.layout == 'C'

    def check_args(a, b, out):
        m, k = a.shape
        _k, n = b.shape
        if k != _k:
            raise ValueError('incompatible array sizes for np.dot(a, b) (matrix * matrix)')
        if out.shape != (m, n):
            raise ValueError('incompatible output array size for np.dot(a, b, out) (matrix * matrix)')
    context.compile_internal(builder, check_args, signature(types.none, *sig.args), args)
    check_c_int(context, builder, m)
    check_c_int(context, builder, k)
    check_c_int(context, builder, n)
    x_data = x.data
    y_data = y.data
    out_data = out.data
    zero = context.get_constant(types.intp, 0)
    both_empty = builder.icmp_signed('==', k, zero)
    x_empty = builder.icmp_signed('==', m, zero)
    y_empty = builder.icmp_signed('==', n, zero)
    is_empty = builder.or_(both_empty, builder.or_(x_empty, y_empty))
    with builder.if_else(is_empty, likely=False) as (empty, nonempty):
        with empty:
            cgutils.memset(builder, out.data, builder.mul(out.itemsize, out.nitems), 0)
        with nonempty:
            one = context.get_constant(types.intp, 1)
            is_left_vec = builder.icmp_signed('==', m, one)
            is_right_vec = builder.icmp_signed('==', n, one)
            with builder.if_else(is_right_vec) as (r_vec, r_mat):
                with r_vec:
                    with builder.if_else(is_left_vec) as (v_v, m_v):
                        with v_v:
                            call_xxdot(context, builder, False, dtype, k, x_data, y_data, out_data)
                        with m_v:
                            do_trans = xty.layout == outty.layout
                            call_xxgemv(context, builder, do_trans, xty, x_shapes, x_data, y_data, out_data)
                with r_mat:
                    with builder.if_else(is_left_vec) as (v_m, m_m):
                        with v_m:
                            do_trans = yty.layout != outty.layout
                            call_xxgemv(context, builder, do_trans, yty, y_shapes, y_data, x_data, out_data)
                        with m_m:
                            call_xxgemm(context, builder, xty, x_shapes, x_data, yty, y_shapes, y_data, outty, out_shapes, out_data)
    return impl_ret_borrowed(context, builder, sig.return_type, out._getvalue())
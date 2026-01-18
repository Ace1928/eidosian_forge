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
def dot_3_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix, out)
    np.dot(matrix, vector, out)
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
    if xty.ndim < yty.ndim:
        mty = yty
        m_shapes = y_shapes
        v_shape = x_shapes[0]
        lda = m_shapes[1]
        do_trans = yty.layout == 'F'
        m_data, v_data = (y.data, x.data)
        check_args = dot_3_vm_check_args
    else:
        mty = xty
        m_shapes = x_shapes
        v_shape = y_shapes[0]
        lda = m_shapes[0]
        do_trans = xty.layout == 'C'
        m_data, v_data = (x.data, y.data)
        check_args = dot_3_mv_check_args
    context.compile_internal(builder, check_args, signature(types.none, *sig.args), args)
    for val in m_shapes:
        check_c_int(context, builder, val)
    zero = context.get_constant(types.intp, 0)
    both_empty = builder.icmp_signed('==', v_shape, zero)
    matrix_empty = builder.icmp_signed('==', lda, zero)
    is_empty = builder.or_(both_empty, matrix_empty)
    with builder.if_else(is_empty, likely=False) as (empty, nonempty):
        with empty:
            cgutils.memset(builder, out.data, builder.mul(out.itemsize, out.nitems), 0)
        with nonempty:
            call_xxgemv(context, builder, do_trans, mty, m_shapes, m_data, v_data, out.data)
    return impl_ret_borrowed(context, builder, sig.return_type, out._getvalue())
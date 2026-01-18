import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def geam(transa, transb, alpha, a, beta, b, out=None):
    """Computes alpha * op(a) + beta * op(b)

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'H'.
    op(b) = b if transb is 'N', op(b) = b.T if transb is 'T',
    op(b) = b.T.conj() if transb is 'H'.
    """
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgeam
    elif dtype == 'd':
        func = cublas.dgeam
    elif dtype == 'F':
        func = cublas.cgeam
    elif dtype == 'D':
        func = cublas.zgeam
    else:
        raise TypeError('invalid dtype')
    transa = _trans_to_cublas_op(transa)
    transb = _trans_to_cublas_op(transb)
    if transa == cublas.CUBLAS_OP_N:
        m, n = a.shape
    else:
        n, m = a.shape
    if transb == cublas.CUBLAS_OP_N:
        assert b.shape == (m, n)
    else:
        assert b.shape == (n, m)
    if out is None:
        out = cupy.empty((m, n), dtype=dtype, order='F')
    else:
        assert out.ndim == 2
        assert out.shape == (m, n)
        assert out.dtype == dtype
    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cupy.ndarray) or isinstance(beta, cupy.ndarray):
        if not isinstance(alpha, cupy.ndarray):
            alpha = cupy.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cupy.ndarray):
            beta = cupy.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    lda, transa = _decide_ld_and_trans(a, transa)
    ldb, transb = _decide_ld_and_trans(b, transb)
    if not (lda is None or ldb is None):
        if out._f_contiguous:
            try:
                func(handle, transa, transb, m, n, alpha_ptr, a.data.ptr, lda, beta_ptr, b.data.ptr, ldb, out.data.ptr, m)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
        elif out._c_contiguous:
            try:
                func(handle, 1 - transa, 1 - transb, n, m, alpha_ptr, a.data.ptr, lda, beta_ptr, b.data.ptr, ldb, out.data.ptr, n)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
    a, lda = _change_order_if_necessary(a, lda)
    b, ldb = _change_order_if_necessary(b, ldb)
    c = out
    if not out._f_contiguous:
        c = out.copy(order='F')
    try:
        func(handle, transa, transb, m, n, alpha_ptr, a.data.ptr, lda, beta_ptr, b.data.ptr, ldb, c.data.ptr, m)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if not out._f_contiguous:
        _core.elementwise_copy(c, out)
    return out
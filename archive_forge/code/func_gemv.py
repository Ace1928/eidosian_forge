import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def gemv(transa, alpha, a, x, beta, y):
    """Computes y = alpha * op(a) @ x + beta * y

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'H'.

    Note: ''y'' will be updated.
    """
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgemv
    elif dtype == 'd':
        func = cublas.dgemv
    elif dtype == 'F':
        func = cublas.cgemv
    elif dtype == 'D':
        func = cublas.zgemv
    else:
        raise TypeError('invalid dtype')
    assert a.ndim == 2
    assert x.ndim == y.ndim == 1
    assert a.dtype == x.dtype == y.dtype
    m, n = a.shape
    transa = _trans_to_cublas_op(transa)
    if transa == cublas.CUBLAS_OP_N:
        xlen, ylen = (n, m)
    else:
        xlen, ylen = (m, n)
    assert x.shape[0] == xlen
    assert y.shape[0] == ylen
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
    try:
        if a._f_contiguous:
            func(handle, transa, m, n, alpha_ptr, a.data.ptr, m, x.data.ptr, 1, beta_ptr, y.data.ptr, 1)
        elif a._c_contiguous and transa != cublas.CUBLAS_OP_C:
            if transa == cublas.CUBLAS_OP_N:
                transa = cublas.CUBLAS_OP_T
            else:
                transa = cublas.CUBLAS_OP_N
            func(handle, transa, n, m, alpha_ptr, a.data.ptr, n, x.data.ptr, 1, beta_ptr, y.data.ptr, 1)
        else:
            a = a.copy(order='F')
            func(handle, transa, m, n, alpha_ptr, a.data.ptr, m, x.data.ptr, 1, beta_ptr, y.data.ptr, 1)
    finally:
        cublas.setPointerMode(handle, orig_mode)
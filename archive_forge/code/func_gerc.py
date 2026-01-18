import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def gerc(alpha, x, y, a):
    """Computes a += alpha * x @ y.T.conj()

    Note: ''a'' will be updated.
    """
    dtype = a.dtype.char
    if dtype in 'fd':
        return ger(alpha, x, y, a)
    elif dtype == 'F':
        func = cublas.cgerc
    elif dtype == 'D':
        func = cublas.zgerc
    else:
        raise TypeError('invalid dtype')
    assert a.ndim == 2
    assert x.ndim == y.ndim == 1
    assert a.dtype == x.dtype == y.dtype
    m, n = a.shape
    assert x.shape[0] == m
    assert y.shape[0] == n
    handle = device.get_cublas_handle()
    alpha, alpha_ptr, orig_mode = _setup_scalar_ptr(handle, alpha, dtype)
    x_ptr, y_ptr = (x.data.ptr, y.data.ptr)
    try:
        if a._f_contiguous:
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, a.data.ptr, m)
        else:
            aa = a.copy(order='F')
            func(handle, m, n, alpha_ptr, x_ptr, 1, y_ptr, 1, aa.data.ptr, m)
            _core.elementwise_copy(aa, a)
    finally:
        cublas.setPointerMode(handle, orig_mode)
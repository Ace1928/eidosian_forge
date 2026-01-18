import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def scal(a, x):
    """Computes x *= a.

    (*) x will be updated.
    """
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))
    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.sscal
    elif dtype == 'd':
        func = cublas.dscal
    elif dtype == 'F':
        func = cublas.cscal
    elif dtype == 'D':
        func = cublas.zscal
    else:
        raise TypeError('invalid dtype')
    handle = device.get_cublas_handle()
    a, a_ptr, orig_mode = _setup_scalar_ptr(handle, a, dtype)
    try:
        func(handle, x.size, a_ptr, x.data.ptr, 1)
    finally:
        cublas.setPointerMode(handle, orig_mode)
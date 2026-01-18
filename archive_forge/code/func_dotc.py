import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def dotc(x, y, out=None):
    """Computes the dot product of x.conj() and y."""
    dtype = x.dtype.char
    if dtype in 'fd':
        return dot(x, y, out=out)
    elif dtype == 'F':
        func = cublas.cdotc
    elif dtype == 'D':
        func = cublas.zdotc
    else:
        raise TypeError('invalid dtype')
    _check_two_vectors(x, y)
    handle = device.get_cublas_handle()
    result_dtype = dtype
    result_ptr, result, orig_mode = _setup_result_ptr(handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, y.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if out is None:
        out = result
    elif out.dtype != result_dtype:
        _core.elementwise_copy(result, out)
    return out
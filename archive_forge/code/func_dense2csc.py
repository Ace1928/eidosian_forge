import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
def dense2csc(x):
    """Converts a dense matrix in CSC format.

    Args:
        x (cupy.ndarray): A matrix to be converted.

    Returns:
        cupyx.scipy.sparse.csc_matrix: A converted matrix.

    """
    if not check_availability('dense2csc'):
        raise RuntimeError('dense2csc is not available.')
    assert x.ndim == 2
    x = _cupy.asfortranarray(x)
    nnz = _numpy.empty((), dtype='i')
    handle = _device.get_cusparse_handle()
    m, n = x.shape
    descr = MatDescriptor.create()
    nnz_per_col = _cupy.empty(m, 'i')
    _call_cusparse('nnz', x.dtype, handle, _cusparse.CUSPARSE_DIRECTION_COLUMN, m, n, descr.descriptor, x.data.ptr, m, nnz_per_col.data.ptr, nnz.ctypes.data)
    nnz = int(nnz)
    data = _cupy.empty(nnz, x.dtype)
    indptr = _cupy.empty(n + 1, 'i')
    indices = _cupy.empty(nnz, 'i')
    _call_cusparse('dense2csc', x.dtype, handle, m, n, descr.descriptor, x.data.ptr, m, nnz_per_col.data.ptr, data.data.ptr, indices.data.ptr, indptr.data.ptr)
    csc = cupyx.scipy.sparse.csc_matrix((data, indices, indptr), shape=x.shape)
    csc._has_canonical_format = True
    return csc
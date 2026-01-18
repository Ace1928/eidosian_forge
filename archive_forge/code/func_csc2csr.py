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
def csc2csr(x):
    if not check_availability('csc2csr'):
        raise RuntimeError('csr2csc is not available.')
    handle = _device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = _cupy.empty(nnz, x.dtype)
    indices = _cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = _cupy.zeros(m + 1, 'i')
    else:
        indptr = _cupy.empty(m + 1, 'i')
        _call_cusparse('csr2csc', x.dtype, handle, n, m, nnz, x.data.data.ptr, x.indptr.data.ptr, x.indices.data.ptr, data.data.ptr, indices.data.ptr, indptr.data.ptr, _cusparse.CUSPARSE_ACTION_NUMERIC, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr_matrix((data, indices, indptr), shape=x.shape)
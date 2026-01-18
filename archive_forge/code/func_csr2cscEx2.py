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
def csr2cscEx2(x):
    if not check_availability('csr2cscEx2'):
        raise RuntimeError('csr2cscEx2 is not available.')
    handle = _device.get_cusparse_handle()
    m, n = x.shape
    nnz = x.nnz
    data = _cupy.empty(nnz, x.dtype)
    indices = _cupy.empty(nnz, 'i')
    if nnz == 0:
        indptr = _cupy.zeros(n + 1, 'i')
    else:
        indptr = _cupy.empty(n + 1, 'i')
        x_dtype = _dtype.to_cuda_dtype(x.dtype)
        action = _cusparse.CUSPARSE_ACTION_NUMERIC
        ibase = _cusparse.CUSPARSE_INDEX_BASE_ZERO
        algo = _cusparse.CUSPARSE_CSR2CSC_ALG1
        buffer_size = _cusparse.csr2cscEx2_bufferSize(handle, m, n, nnz, x.data.data.ptr, x.indptr.data.ptr, x.indices.data.ptr, data.data.ptr, indptr.data.ptr, indices.data.ptr, x_dtype, action, ibase, algo)
        buffer = _cupy.empty(buffer_size, _numpy.int8)
        _cusparse.csr2cscEx2(handle, m, n, nnz, x.data.data.ptr, x.indptr.data.ptr, x.indices.data.ptr, data.data.ptr, indptr.data.ptr, indices.data.ptr, x_dtype, action, ibase, algo, buffer.data.ptr)
    return cupyx.scipy.sparse.csc_matrix((data, indices, indptr), shape=x.shape)
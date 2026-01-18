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
def coo2csr(x):
    handle = _device.get_cusparse_handle()
    m = x.shape[0]
    nnz = x.nnz
    if nnz == 0:
        indptr = _cupy.zeros(m + 1, 'i')
    else:
        indptr = _cupy.empty(m + 1, 'i')
        _cusparse.xcoo2csr(handle, x.row.data.ptr, nnz, m, indptr.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    return cupyx.scipy.sparse.csr_matrix((x.data, x.col, indptr), shape=x.shape)
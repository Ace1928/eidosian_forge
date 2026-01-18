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
def cscsort(x):
    """Sorts indices of CSC-matrix in place.

    Args:
        x (cupyx.scipy.sparse.csc_matrix): A sparse matrix to sort.

    """
    if not check_availability('cscsort'):
        raise RuntimeError('cscsort is not available.')
    nnz = x.nnz
    if nnz == 0:
        return
    handle = _device.get_cusparse_handle()
    m, n = x.shape
    buffer_size = _cusparse.xcscsort_bufferSizeExt(handle, m, n, nnz, x.indptr.data.ptr, x.indices.data.ptr)
    buf = _cupy.empty(buffer_size, 'b')
    P = _cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    _cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    _cusparse.xcscsort(handle, m, n, nnz, x._descr.descriptor, x.indptr.data.ptr, x.indices.data.ptr, P.data.ptr, buf.data.ptr)
    if check_availability('gthr'):
        _call_cusparse('gthr', x.dtype, handle, nnz, data_orig.data.ptr, x.data.data.ptr, P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    else:
        desc_x = SpVecDescriptor.create(P, x.data)
        desc_y = DnVecDescriptor.create(data_orig)
        _cusparse.gather(handle, desc_y.desc, desc_x.desc)
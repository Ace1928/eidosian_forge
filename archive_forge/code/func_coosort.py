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
def coosort(x, sort_by='r'):
    """Sorts indices of COO-matrix in place.

    Args:
        x (cupyx.scipy.sparse.coo_matrix): A sparse matrix to sort.
        sort_by (str): Sort the indices by row ('r', default) or column ('c').

    """
    if not check_availability('coosort'):
        raise RuntimeError('coosort is not available.')
    nnz = x.nnz
    if nnz == 0:
        return
    handle = _device.get_cusparse_handle()
    m, n = x.shape
    buffer_size = _cusparse.xcoosort_bufferSizeExt(handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr)
    buf = _cupy.empty(buffer_size, 'b')
    P = _cupy.empty(nnz, 'i')
    data_orig = x.data.copy()
    _cusparse.createIdentityPermutation(handle, nnz, P.data.ptr)
    if sort_by == 'r':
        _cusparse.xcoosortByRow(handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr, P.data.ptr, buf.data.ptr)
    elif sort_by == 'c':
        _cusparse.xcoosortByColumn(handle, m, n, nnz, x.row.data.ptr, x.col.data.ptr, P.data.ptr, buf.data.ptr)
    else:
        raise ValueError("sort_by must be either 'r' or 'c'")
    if check_availability('gthr'):
        _call_cusparse('gthr', x.dtype, handle, nnz, data_orig.data.ptr, x.data.data.ptr, P.data.ptr, _cusparse.CUSPARSE_INDEX_BASE_ZERO)
    else:
        desc_x = SpVecDescriptor.create(P, x.data)
        desc_y = DnVecDescriptor.create(data_orig)
        _cusparse.gather(handle, desc_y.desc, desc_x.desc)
    if sort_by == 'c':
        x._has_canonical_format = False
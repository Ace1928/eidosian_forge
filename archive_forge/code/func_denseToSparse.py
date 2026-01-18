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
def denseToSparse(x, format='csr'):
    """Converts a dense matrix into a CSR, CSC or COO format.

    Args:
        x (cupy.ndarray): A matrix to be converted.
        format (str): Format of converted matrix. It must be either 'csr',
            'csc' or 'coo'.

    Returns:
        cupyx.scipy.sparse.spmatrix: A converted sparse matrix.

    """
    if not check_availability('denseToSparse'):
        raise RuntimeError('denseToSparse is not available.')
    assert x.ndim == 2
    assert x.dtype.char in 'fdFD'
    x = _cupy.asfortranarray(x)
    desc_x = DnMatDescriptor.create(x)
    if format == 'csr':
        y = cupyx.scipy.sparse.csr_matrix(x.shape, dtype=x.dtype)
    elif format == 'csc':
        y = cupyx.scipy.sparse.csc_matrix(x.shape, dtype=x.dtype)
    elif format == 'coo':
        y = cupyx.scipy.sparse.coo_matrix(x.shape, dtype=x.dtype)
    else:
        raise TypeError('unsupported format (actual: {})'.format(format))
    desc_y = SpMatDescriptor.create(y)
    algo = _cusparse.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT
    handle = _device.get_cusparse_handle()
    buff_size = _cusparse.denseToSparse_bufferSize(handle, desc_x.desc, desc_y.desc, algo)
    buff = _cupy.empty(buff_size, _cupy.int8)
    _cusparse.denseToSparse_analysis(handle, desc_x.desc, desc_y.desc, algo, buff.data.ptr)
    num_rows_tmp = _numpy.array(0, dtype='int64')
    num_cols_tmp = _numpy.array(0, dtype='int64')
    nnz = _numpy.array(0, dtype='int64')
    _cusparse.spMatGetSize(desc_y.desc, num_rows_tmp.ctypes.data, num_cols_tmp.ctypes.data, nnz.ctypes.data)
    nnz = int(nnz)
    if _runtime.is_hip:
        if nnz == 0:
            raise ValueError('hipSPARSE currently cannot handle sparse matrices with null ptrs')
    if format == 'csr':
        indptr = y.indptr
        indices = _cupy.empty(nnz, 'i')
        data = _cupy.empty(nnz, x.dtype)
        y = cupyx.scipy.sparse.csr_matrix((data, indices, indptr), shape=x.shape)
    elif format == 'csc':
        indptr = y.indptr
        indices = _cupy.empty(nnz, 'i')
        data = _cupy.empty(nnz, x.dtype)
        y = cupyx.scipy.sparse.csc_matrix((data, indices, indptr), shape=x.shape)
    elif format == 'coo':
        row = _cupy.zeros(nnz, 'i')
        col = _cupy.zeros(nnz, 'i')
        data = _cupy.empty(nnz, x.dtype)
        y = cupyx.scipy.sparse.coo_matrix((data, (row, col)), shape=x.shape)
    desc_y = SpMatDescriptor.create(y)
    _cusparse.denseToSparse_convert(handle, desc_x.desc, desc_y.desc, algo, buff.data.ptr)
    y._has_canonical_format = True
    return y
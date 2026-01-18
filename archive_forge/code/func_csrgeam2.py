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
def csrgeam2(a, b, alpha=1, beta=1):
    """Matrix-matrix addition.

    .. math::
        C = \\alpha A + \\beta B

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        alpha (float): Coefficient for A.
        beta (float): Coefficient for B.

    Returns:
        cupyx.scipy.sparse.csr_matrix: Result matrix.

    """
    if not check_availability('csrgeam2'):
        raise RuntimeError('csrgeam2 is not available.')
    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))
    assert a.has_canonical_format
    assert b.has_canonical_format
    if a.shape != b.shape:
        raise ValueError('inconsistent shapes')
    handle = _device.get_cusparse_handle()
    m, n = a.shape
    a, b = _cast_common_type(a, b)
    nnz = _numpy.empty((), 'i')
    _cusparse.setPointerMode(handle, _cusparse.CUSPARSE_POINTER_MODE_HOST)
    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
    c_descr = MatDescriptor.create()
    c_indptr = _cupy.empty(m + 1, 'i')
    null_ptr = 0
    buff_size = _call_cusparse('csrgeam2_bufferSizeExt', a.dtype, handle, m, n, alpha.data, a._descr.descriptor, a.nnz, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, beta.data, b._descr.descriptor, b.nnz, b.data.data.ptr, b.indptr.data.ptr, b.indices.data.ptr, c_descr.descriptor, null_ptr, c_indptr.data.ptr, null_ptr)
    buff = _cupy.empty(buff_size, _numpy.int8)
    _cusparse.xcsrgeam2Nnz(handle, m, n, a._descr.descriptor, a.nnz, a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr, b.indices.data.ptr, c_descr.descriptor, c_indptr.data.ptr, nnz.ctypes.data, buff.data.ptr)
    c_indices = _cupy.empty(int(nnz), 'i')
    c_data = _cupy.empty(int(nnz), a.dtype)
    _call_cusparse('csrgeam2', a.dtype, handle, m, n, alpha.data, a._descr.descriptor, a.nnz, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, beta.data, b._descr.descriptor, b.nnz, b.data.data.ptr, b.indptr.data.ptr, b.indices.data.ptr, c_descr.descriptor, c_data.data.ptr, c_indptr.data.ptr, c_indices.data.ptr, buff.data.ptr)
    c = cupyx.scipy.sparse.csr_matrix((c_data, c_indices, c_indptr), shape=a.shape)
    c._has_canonical_format = True
    return c
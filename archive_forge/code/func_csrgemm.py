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
def csrgemm(a, b, transa=False, transb=False):
    """Matrix-matrix product for CSR-matrix.

    math::
       C = op(A) op(B),

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        transa (bool): If ``True``, transpose of A is used.
        transb (bool): If ``True``, transpose of B is used.

    Returns:
        cupyx.scipy.sparse.csr_matrix: Calculated C.

    """
    if not check_availability('csrgemm'):
        raise RuntimeError('csrgemm is not available.')
    assert a.ndim == b.ndim == 2
    assert a.has_canonical_format
    assert b.has_canonical_format
    a_shape = a.shape if not transa else a.shape[::-1]
    b_shape = b.shape if not transb else b.shape[::-1]
    if a_shape[1] != b_shape[0]:
        raise ValueError('dimension mismatch')
    handle = _device.get_cusparse_handle()
    m, k = a_shape
    n = b_shape[1]
    a, b = _cast_common_type(a, b)
    if a.nnz == 0 or b.nnz == 0:
        return cupyx.scipy.sparse.csr_matrix((m, n), dtype=a.dtype)
    op_a = _transpose_flag(transa)
    op_b = _transpose_flag(transb)
    nnz = _numpy.empty((), 'i')
    _cusparse.setPointerMode(handle, _cusparse.CUSPARSE_POINTER_MODE_HOST)
    c_descr = MatDescriptor.create()
    c_indptr = _cupy.empty(m + 1, 'i')
    _cusparse.xcsrgemmNnz(handle, op_a, op_b, m, n, k, a._descr.descriptor, a.nnz, a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr, b.indices.data.ptr, c_descr.descriptor, c_indptr.data.ptr, nnz.ctypes.data)
    c_indices = _cupy.empty(int(nnz), 'i')
    c_data = _cupy.empty(int(nnz), a.dtype)
    _call_cusparse('csrgemm', a.dtype, handle, op_a, op_b, m, n, k, a._descr.descriptor, a.nnz, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz, b.data.data.ptr, b.indptr.data.ptr, b.indices.data.ptr, c_descr.descriptor, c_data.data.ptr, c_indptr.data.ptr, c_indices.data.ptr)
    c = cupyx.scipy.sparse.csr_matrix((c_data, c_indices, c_indptr), shape=(m, n))
    c._has_canonical_format = True
    return c
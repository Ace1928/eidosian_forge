import operator
import warnings
import numpy
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util
def multiply_by_csr(a, b):
    check_shape_for_pointwise_op(a.shape, b.shape)
    a_m, a_n = a.shape
    b_m, b_n = b.shape
    m, n = (max(a_m, b_m), max(a_n, b_n))
    a_nnz = a.nnz * (m // a_m) * (n // a_n)
    b_nnz = b.nnz * (m // b_m) * (n // b_n)
    if a_nnz > b_nnz:
        return multiply_by_csr(b, a)
    c_nnz = a_nnz
    dtype = numpy.promote_types(a.dtype, b.dtype)
    c_data = cupy.empty(c_nnz, dtype=dtype)
    c_indices = cupy.empty(c_nnz, dtype=a.indices.dtype)
    if m > a_m:
        if n > a_n:
            c_indptr = cupy.arange(0, c_nnz + 1, n, dtype=a.indptr.dtype)
        else:
            c_indptr = cupy.arange(0, c_nnz + 1, a.nnz, dtype=a.indptr.dtype)
    else:
        c_indptr = a.indptr.copy()
        if n > a_n:
            c_indptr *= n
    flags = cupy.zeros(c_nnz + 1, dtype=a.indices.dtype)
    nnz_each_row = cupy.zeros(m + 1, dtype=a.indptr.dtype)
    cupy_multiply_by_csr_step1()(a.data, a.indptr, a.indices, a_m, a_n, b.data, b.indptr, b.indices, b_m, b_n, c_indptr, m, n, c_data, c_indices, flags, nnz_each_row)
    flags = cupy.cumsum(flags, dtype=a.indptr.dtype)
    d_indptr = cupy.cumsum(nnz_each_row, dtype=a.indptr.dtype)
    d_nnz = int(d_indptr[-1])
    d_data = cupy.empty(d_nnz, dtype=dtype)
    d_indices = cupy.empty(d_nnz, dtype=a.indices.dtype)
    cupy_multiply_by_csr_step2()(c_data, c_indices, flags, d_data, d_indices)
    return csr_matrix((d_data, d_indices, d_indptr), shape=(m, n))
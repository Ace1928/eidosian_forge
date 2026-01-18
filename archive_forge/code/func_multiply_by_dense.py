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
def multiply_by_dense(sp, dn):
    check_shape_for_pointwise_op(sp.shape, dn.shape)
    sp_m, sp_n = sp.shape
    dn_m, dn_n = dn.shape
    m, n = (max(sp_m, dn_m), max(sp_n, dn_n))
    nnz = sp.nnz * (m // sp_m) * (n // sp_n)
    dtype = numpy.promote_types(sp.dtype, dn.dtype)
    data = cupy.empty(nnz, dtype=dtype)
    indices = cupy.empty(nnz, dtype=sp.indices.dtype)
    if m > sp_m:
        if n > sp_n:
            indptr = cupy.arange(0, nnz + 1, n, dtype=sp.indptr.dtype)
        else:
            indptr = cupy.arange(0, nnz + 1, sp.nnz, dtype=sp.indptr.dtype)
    else:
        indptr = sp.indptr.copy()
        if n > sp_n:
            indptr *= n
    cupy_multiply_by_dense()(sp.data, sp.indptr, sp.indices, sp_m, sp_n, dn, dn_m, dn_n, indptr, m, n, data, indices)
    return csr_matrix((data, indices, indptr), shape=(m, n))
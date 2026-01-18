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
def spmv(a, x, y=None, alpha=1, beta=0, transa=False):
    """Multiplication of sparse matrix and dense vector.

    .. math::

        y = \\alpha * op(A) x + \\beta * y

    Args:
        a (cupyx.scipy.sparse.csr_matrix, csc_matrix or coo_matrix):
            Sparse matrix A
        x (cupy.ndarray): Dense vector x
        y (cupy.ndarray or None): Dense vector y
        alpha (scalar): Coefficent
        beta (scalar): Coefficent
        transa (bool): If ``True``, op(A) = transpose of A.

    Returns:
        cupy.ndarray
    """
    if not check_availability('spmv'):
        raise RuntimeError('spmv is not available.')
    if isinstance(a, cupyx.scipy.sparse.csc_matrix):
        aT = a.T
        if not isinstance(aT, cupyx.scipy.sparse.csr_matrix):
            msg = 'aT must be csr_matrix (actual: {})'.format(type(aT))
            raise TypeError(msg)
        a = aT
        transa = not transa
    if not isinstance(a, (cupyx.scipy.sparse.csr_matrix, cupyx.scipy.sparse.coo_matrix)):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    a_shape = a.shape if not transa else a.shape[::-1]
    if a_shape[1] != len(x):
        raise ValueError('dimension mismatch')
    assert a.has_canonical_format
    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    if y is None:
        y = _cupy.zeros(m, a.dtype)
    elif len(y) != m:
        raise ValueError('dimension mismatch')
    if a.nnz == 0:
        y.fill(0)
        return y
    desc_a = SpMatDescriptor.create(a)
    desc_x = DnVecDescriptor.create(x)
    desc_y = DnVecDescriptor.create(y)
    handle = _device.get_cusparse_handle()
    op_a = _transpose_flag(transa)
    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
    cuda_dtype = _dtype.to_cuda_dtype(a.dtype)
    alg = _cusparse.CUSPARSE_MV_ALG_DEFAULT
    buff_size = _cusparse.spMV_bufferSize(handle, op_a, alpha.data, desc_a.desc, desc_x.desc, beta.data, desc_y.desc, cuda_dtype, alg)
    buff = _cupy.empty(buff_size, _cupy.int8)
    _cusparse.spMV(handle, op_a, alpha.data, desc_a.desc, desc_x.desc, beta.data, desc_y.desc, cuda_dtype, alg, buff.data.ptr)
    return y
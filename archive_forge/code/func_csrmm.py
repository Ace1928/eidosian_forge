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
def csrmm(a, b, c=None, alpha=1, beta=0, transa=False):
    """Matrix-matrix product for a CSR-matrix and a dense matrix.

    .. math::

       C = \\alpha o_a(A) B + \\beta C,

    where :math:`o_a` is a transpose function when ``transa`` is ``True`` and
    is an identity function otherwise.

    Args:
        a (cupyx.scipy.sparse.csr): Sparse matrix A.
        b (cupy.ndarray): Dense matrix B. It must be F-contiguous.
        c (cupy.ndarray or None): Dense matrix C. It must be F-contiguous.
        alpha (float): Coefficient for AB.
        beta (float): Coefficient for C.
        transa (bool): If ``True``, transpose of A is used.

    Returns:
        cupy.ndarray: Calculated C.

    """
    if not check_availability('csrmm'):
        raise RuntimeError('csrmm is not available.')
    assert a.ndim == b.ndim == 2
    assert b.flags.f_contiguous
    assert c is None or c.flags.f_contiguous
    a_shape = a.shape if not transa else a.shape[::-1]
    if a_shape[1] != b.shape[0]:
        raise ValueError('dimension mismatch')
    handle = _device.get_cusparse_handle()
    m, k = a_shape
    n = b.shape[1]
    a, b, c = _cast_common_type(a, b, c)
    if c is None:
        c = _cupy.zeros((m, n), a.dtype, 'F')
    ldb = k
    ldc = m
    alpha = _numpy.array(alpha, a.dtype).ctypes
    beta = _numpy.array(beta, a.dtype).ctypes
    _call_cusparse('csrmm', a.dtype, handle, _transpose_flag(transa), a.shape[0], n, a.shape[1], a.nnz, alpha.data, a._descr.descriptor, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, b.data.ptr, ldb, beta.data, c.data.ptr, ldc)
    return c
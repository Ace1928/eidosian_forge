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
def csrmvEx(a, x, y=None, alpha=1, beta=0, merge_path=True):
    """Matrix-vector product for a CSR-matrix and a dense vector.

    .. math::

       y = \\alpha * A x + \\beta y,

    Args:
        a (cupyx.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y. It must be F-contiguous.
        alpha (float): Coefficient for x.
        beta (float): Coefficient for y.
        merge_path (bool): If ``True``, merge path algorithm is used.

        All pointers must be aligned with 128 bytes.

    Returns:
        cupy.ndarray: Calculated ``y``.

    """
    if not check_availability('csrmvEx'):
        raise RuntimeError('csrmvEx is not available.')
    assert y is None or y.flags.f_contiguous
    if a.shape[1] != len(x):
        raise ValueError('dimension mismatch')
    handle = _device.get_cusparse_handle()
    m, n = a.shape
    a, x, y = _cast_common_type(a, x, y)
    dtype = a.dtype
    if y is None:
        y = _cupy.zeros(m, dtype)
    datatype = _dtype.to_cuda_dtype(dtype)
    algmode = _cusparse.CUSPARSE_ALG_MERGE_PATH if merge_path else _cusparse.CUSPARSE_ALG_NAIVE
    transa_flag = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    alpha = _numpy.array(alpha, dtype).ctypes
    beta = _numpy.array(beta, dtype).ctypes
    assert csrmvExIsAligned(a, x, y)
    bufferSize = _cusparse.csrmvEx_bufferSize(handle, algmode, transa_flag, a.shape[0], a.shape[1], a.nnz, alpha.data, datatype, a._descr.descriptor, a.data.data.ptr, datatype, a.indptr.data.ptr, a.indices.data.ptr, x.data.ptr, datatype, beta.data, datatype, y.data.ptr, datatype, datatype)
    buf = _cupy.empty(bufferSize, 'b')
    assert buf.data.ptr % 128 == 0
    _cusparse.csrmvEx(handle, algmode, transa_flag, a.shape[0], a.shape[1], a.nnz, alpha.data, datatype, a._descr.descriptor, a.data.data.ptr, datatype, a.indptr.data.ptr, a.indices.data.ptr, x.data.ptr, datatype, beta.data, datatype, y.data.ptr, datatype, datatype, buf.data.ptr)
    return y
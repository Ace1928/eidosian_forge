import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusolver as _cusolver
from cupy.cuda import device as _device
import cupyx.cusolver
def posv(a, b):
    """Solve the linear equations A x = b via Cholesky factorization of A,
    where A is a real symmetric or complex Hermitian positive-definite matrix.

    If matrix ``A`` is not positive definite, Cholesky factorization fails
    and it raises an error.

    Note: For batch input, NRHS > 1 is not currently supported.

    Args:
        a (cupy.ndarray): Array of real symmetric or complex hermitian
            matrices with dimension (..., N, N).
        b (cupy.ndarray): right-hand side (..., N) or (..., N, NRHS).
    Returns:
        x (cupy.ndarray): The solution (shape matches b).
    """
    _util = _cupy.linalg._util
    _util._assert_cupy_array(a, b)
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)
    if a.ndim > 2:
        return _batched_posv(a, b)
    dtype = _numpy.promote_types(a.dtype, b.dtype)
    dtype = _numpy.promote_types(dtype, 'f')
    if dtype == 'f':
        potrf = _cusolver.spotrf
        potrf_bufferSize = _cusolver.spotrf_bufferSize
        potrs = _cusolver.spotrs
    elif dtype == 'd':
        potrf = _cusolver.dpotrf
        potrf_bufferSize = _cusolver.dpotrf_bufferSize
        potrs = _cusolver.dpotrs
    elif dtype == 'F':
        potrf = _cusolver.cpotrf
        potrf_bufferSize = _cusolver.cpotrf_bufferSize
        potrs = _cusolver.cpotrs
    elif dtype == 'D':
        potrf = _cusolver.zpotrf
        potrf_bufferSize = _cusolver.zpotrf_bufferSize
        potrs = _cusolver.zpotrs
    else:
        msg = 'dtype must be float32, float64, complex64 or complex128 (actual: {})'.format(a.dtype)
        raise ValueError(msg)
    a = a.astype(dtype, order='F', copy=True)
    lda, n = a.shape
    handle = _device.get_cusolver_handle()
    uplo = _cublas.CUBLAS_FILL_MODE_LOWER
    dev_info = _cupy.empty(1, dtype=_numpy.int32)
    worksize = potrf_bufferSize(handle, uplo, n, a.data.ptr, lda)
    workspace = _cupy.empty(worksize, dtype=dtype)
    potrf(handle, uplo, n, a.data.ptr, lda, workspace.data.ptr, worksize, dev_info.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(potrf, dev_info)
    b_shape = b.shape
    b = b.reshape(n, -1).astype(dtype, order='F', copy=True)
    ldb, nrhs = b.shape
    potrs(handle, uplo, n, nrhs, a.data.ptr, lda, b.data.ptr, ldb, dev_info.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(potrs, dev_info)
    return _cupy.ascontiguousarray(b.reshape(b_shape))
from cupy.random import _generator
from cupy import _util
def multivariate_normal(mean, cov, size=None, check_valid='ignore', tol=1e-08, method='cholesky', dtype=float):
    """Multivariate normal distribution.

    Returns an array of samples drawn from the multivariate normal
    distribution. Its probability density function is defined as

    .. math::
       f(x) = \\frac{1}{(2\\pi|\\Sigma|)^(n/2)}            \\exp\\left(-\\frac{1}{2}            (x-\\mu)^{\\top}\\Sigma^{-1}(x-\\mu)\\right).

    Args:
        mean (1-D array_like, of length N): Mean of the multivariate normal
            distribution :math:`\\mu`.
        cov (2-D array_like, of shape (N, N)): Covariance matrix
            :math:`\\Sigma` of the multivariate normal distribution. It must be
            symmetric and positive-semidefinite for proper sampling.
        size (int or tuple of ints): The shape of the array. If ``None``, a
            zero-dimensional array is generated.
        check_valid ('warn', 'raise', 'ignore'): Behavior when the covariance
            matrix is not positive semidefinite.
        tol (float): Tolerance when checking the singular values in
            covariance matrix.
        method : { 'cholesky', 'eigh', 'svd'}, optional
            The cov input is used to compute a factor matrix A such that
            ``A @ A.T = cov``. This argument is used to select the method
            used to compute the factor matrix A. The default method 'cholesky'
            is the fastest, while 'svd' is the slowest but more robust than
            the fastest method. The method `eigh` uses eigen decomposition to
            compute A and is faster than svd but slower than cholesky.
        dtype: Data type specifier. Only :class:`numpy.float32` and
            :class:`numpy.float64` types are allowed.

    Returns:
        cupy.ndarray: Samples drawn from the multivariate normal distribution.

    .. note:: Default `method` is set to fastest, 'cholesky', unlike numpy
        which defaults to 'svd'. Cholesky decomposition in CuPy will fail
        silently if the input covariance matrix is not positive definite and
        give invalid results, unlike in numpy, where an invalid covariance
        matrix will raise an exception. Setting `check_valid` to 'raise' will
        replicate numpy behavior by checking the input, but will also force
        device synchronization. If validity of input is unknown, setting
        `method` to 'einh' or 'svd' and `check_valid` to 'warn' will use
        cholesky decomposition for positive definite matrices, and fallback to
        the specified `method` for other matrices (i.e., not positive
        semi-definite), and will warn if decomposition is suspect.

    .. seealso:: :func:`numpy.random.multivariate_normal`

    """
    _util.experimental('cupy.random.multivariate_normal')
    rs = _generator.get_random_state()
    return rs.multivariate_normal(mean, cov, size, check_valid, tol, method, dtype)
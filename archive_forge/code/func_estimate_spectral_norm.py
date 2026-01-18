import scipy.linalg._interpolative_backend as _backend
import numpy as np
import sys
def estimate_spectral_norm(A, its=20):
    """
    Estimate spectral norm of a matrix by the randomized power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_snorm` and
        :func:`_backend.idz_snorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        Matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    its : int, optional
        Number of power method iterations.

    Returns
    -------
    float
        Spectral norm estimate.
    """
    from scipy.sparse.linalg import aslinearoperator
    A = aslinearoperator(A)
    m, n = A.shape

    def matvec(x):
        return A.matvec(x)

    def matveca(x):
        return A.rmatvec(x)
    if _is_real(A):
        return _backend.idd_snorm(m, n, matveca, matvec, its=its)
    else:
        return _backend.idz_snorm(m, n, matveca, matvec, its=its)
import scipy.linalg._interpolative as _id
import numpy as np
def iddp_asvd(eps, A):
    """
    Compute SVD of a real matrix to a specified relative precision using random
    sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Left singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Right singular vectors.
    :rtype: :class:`numpy.ndarray`
    :return:
        Singular values.
    :rtype: :class:`numpy.ndarray`
    """
    A = np.asfortranarray(A)
    m, n = A.shape
    n2, winit = _id.idd_frmi(m)
    w = np.empty(max((min(m, n) + 1) * (3 * m + 5 * n + 1) + 25 * min(m, n) ** 2, (2 * n + 1) * (n2 + 1)), order='F')
    k, iU, iV, iS, w, ier = _id.iddp_asvd(eps, A, winit, w)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)
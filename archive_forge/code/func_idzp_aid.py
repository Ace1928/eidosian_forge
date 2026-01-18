import scipy.linalg._interpolative as _id
import numpy as np
def idzp_aid(eps, A):
    """
    Compute ID of a complex matrix to a specified relative precision using
    random sampling.

    :param eps:
        Relative precision.
    :type eps: float
    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`

    :return:
        Rank of ID.
    :rtype: int
    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    A = np.asfortranarray(A)
    m, n = A.shape
    n2, w = idz_frmi(m)
    proj = np.empty(n * (2 * n2 + 1) + n2 + 1, dtype='complex128', order='F')
    k, idx, proj = _id.idzp_aid(eps, A, w, proj)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)
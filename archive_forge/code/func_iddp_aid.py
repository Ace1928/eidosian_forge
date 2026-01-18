import scipy.linalg._interpolative as _id
import numpy as np
def iddp_aid(eps, A):
    """
    Compute ID of a real matrix to a specified relative precision using random
    sampling.

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
    n2, w = idd_frmi(m)
    proj = np.empty(n * (2 * n2 + 1) + n2 + 1, order='F')
    k, idx, proj = _id.iddp_aid(eps, A, w, proj)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)
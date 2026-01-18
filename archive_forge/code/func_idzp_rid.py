import scipy.linalg._interpolative as _id
import numpy as np
def idzp_rid(eps, m, n, matveca):
    """
    Compute ID of a complex matrix to a specified relative precision using
    random matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matveca:
        Function to apply the matrix adjoint to a vector, with call signature
        `y = matveca(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matveca: function

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
    proj = np.empty(m + 1 + 2 * n * (min(m, n) + 1), dtype=np.complex128, order='F')
    k, idx, proj, ier = _id.idzp_rid(eps, m, n, matveca, proj)
    if ier:
        raise _RETCODE_ERROR
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)
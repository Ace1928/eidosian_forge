import scipy.linalg._interpolative as _id
import numpy as np
def iddp_rid(eps, m, n, matvect):
    """
    Compute ID of a real matrix to a specified relative precision using random
    matrix-vector multiplication.

    :param eps:
        Relative precision.
    :type eps: float
    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the matrix transpose to a vector, with call signature
        `y = matvect(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvect: function

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
    proj = np.empty(m + 1 + 2 * n * (min(m, n) + 1), order='F')
    k, idx, proj, ier = _id.iddp_rid(eps, m, n, matvect, proj)
    if ier != 0:
        raise _RETCODE_ERROR
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (k, idx, proj)
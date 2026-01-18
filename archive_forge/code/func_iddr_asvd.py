import scipy.linalg._interpolative as _id
import numpy as np
def iddr_asvd(A, k):
    """
    Compute SVD of a real matrix to a specified rank using random sampling.

    :param A:
        Matrix.
    :type A: :class:`numpy.ndarray`
    :param k:
        Rank of SVD.
    :type k: int

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
    w = np.empty((2 * k + 28) * m + (6 * k + 21) * n + 25 * k ** 2 + 100, order='F')
    w_ = iddr_aidi(m, n, k)
    w[:w_.size] = w_
    U, V, S, ier = _id.iddr_asvd(A, k, w)
    if ier != 0:
        raise _RETCODE_ERROR
    return (U, V, S)
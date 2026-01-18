import scipy.linalg._interpolative as _id
import numpy as np
def idzp_rsvd(eps, m, n, matveca, matvec):
    """
    Compute SVD of a complex matrix to a specified relative precision using
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
    :param matvec:
        Function to apply the matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function

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
    k, iU, iV, iS, w, ier = _id.idzp_rsvd(eps, m, n, matveca, matvec)
    if ier:
        raise _RETCODE_ERROR
    U = w[iU - 1:iU + m * k - 1].reshape((m, k), order='F')
    V = w[iV - 1:iV + n * k - 1].reshape((n, k), order='F')
    S = w[iS - 1:iS + k - 1]
    return (U, V, S)
import scipy.linalg._interpolative as _id
import numpy as np
def idzr_rsvd(m, n, matveca, matvec, k):
    """
    Compute SVD of a complex matrix to a specified rank using random
    matrix-vector multiplication.

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
    U, V, S, ier = _id.idzr_rsvd(m, n, matveca, matvec, k)
    if ier:
        raise _RETCODE_ERROR
    return (U, V, S)
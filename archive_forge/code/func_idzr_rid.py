import scipy.linalg._interpolative as _id
import numpy as np
def idzr_rid(m, n, matveca, k):
    """
    Compute ID of a complex matrix to a specified rank using random
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
    :param k:
        Rank of ID.
    :type k: int

    :return:
        Column index array.
    :rtype: :class:`numpy.ndarray`
    :return:
        Interpolation coefficients.
    :rtype: :class:`numpy.ndarray`
    """
    idx, proj = _id.idzr_rid(m, n, matveca, k)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (idx, proj)
import scipy.linalg._interpolative as _id
import numpy as np
def iddr_rid(m, n, matvect, k):
    """
    Compute ID of a real matrix to a specified rank using random matrix-vector
    multiplication.

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
    idx, proj = _id.iddr_rid(m, n, matvect, k)
    proj = proj[:k * (n - k)].reshape((k, n - k), order='F')
    return (idx, proj)
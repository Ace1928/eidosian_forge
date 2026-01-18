import scipy.linalg._interpolative as _id
import numpy as np
def idz_snorm(m, n, matveca, matvec, its=20):
    """
    Estimate spectral norm of a complex matrix by the randomized power method.

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
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate.
    :rtype: float
    """
    snorm, v = _id.idz_snorm(m, n, matveca, matvec, its)
    return snorm
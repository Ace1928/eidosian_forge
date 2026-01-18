import scipy.linalg._interpolative as _id
import numpy as np
def idd_snorm(m, n, matvect, matvec, its=20):
    """
    Estimate spectral norm of a real matrix by the randomized power method.

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
    snorm, v = _id.idd_snorm(m, n, matvect, matvec, its)
    return snorm
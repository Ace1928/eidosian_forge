import scipy.linalg._interpolative as _id
import numpy as np
def idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its=20):
    """
    Estimate spectral norm of the difference of two real matrices by the
    randomized power method.

    :param m:
        Matrix row dimension.
    :type m: int
    :param n:
        Matrix column dimension.
    :type n: int
    :param matvect:
        Function to apply the transpose of the first matrix to a vector, with
        call signature `y = matvect(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matvect: function
    :param matvect2:
        Function to apply the transpose of the second matrix to a vector, with
        call signature `y = matvect2(x)`, where `x` and `y` are the input and
        output vectors, respectively.
    :type matvect2: function
    :param matvec:
        Function to apply the first matrix to a vector, with call signature
        `y = matvec(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec: function
    :param matvec2:
        Function to apply the second matrix to a vector, with call signature
        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
        respectively.
    :type matvec2: function
    :param its:
        Number of power method iterations.
    :type its: int

    :return:
        Spectral norm estimate of matrix difference.
    :rtype: float
    """
    return _id.idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its)
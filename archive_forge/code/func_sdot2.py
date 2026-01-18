import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def sdot2(x, y):
    """
    Inner product of two block-diagonal symmetric dense 'd' matrices.

    x and y are square dense 'd' matrices, or lists of N square dense 'd' 
    matrices.
    """
    a = 0.0
    if type(x) is matrix:
        n = x.size[0]
        a += blas.dot(x, y, incx=n + 1, incy=n + 1, n=n)
        for j in range(1, n):
            a += 2.0 * blas.dot(x, y, incx=n + 1, incy=n + 1, offsetx=j, offsety=j, n=n - j)
    else:
        for k in range(len(x)):
            n = x[k].size[0]
            a += blas.dot(x[k], y[k], incx=n + 1, incy=n + 1, n=n)
            for j in range(1, n):
                a += 2.0 * blas.dot(x[k], y[k], incx=n + 1, incy=n + 1, offsetx=j, offsety=j, n=n - j)
    return a
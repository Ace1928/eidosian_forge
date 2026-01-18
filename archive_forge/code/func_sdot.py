import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def sdot(x, y, dims, mnl=0):
    """
    Inner product of two vectors in S.
    """
    ind = mnl + dims['l'] + sum(dims['q'])
    a = blas.dot(x, y, n=ind)
    for m in dims['s']:
        a += blas.dot(x, y, offsetx=ind, offsety=ind, incx=m + 1, incy=m + 1, n=m)
        for j in range(1, m):
            a += 2.0 * blas.dot(x, y, incx=m + 1, incy=m + 1, offsetx=ind + j, offsety=ind + j, n=m - j)
        ind += m ** 2
    return a
import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def sinv(x, y, dims, mnl=0):
    """
    The inverse product x := (y o\\ x), when the 's' components of y are 
    diagonal.
    """
    blas.tbsv(y, x, n=mnl + dims['l'], k=0, ldA=1)
    ind = mnl + dims['l']
    for m in dims['q']:
        aa = jnrm2(y, n=m, offset=ind) ** 2
        cc = x[ind]
        dd = blas.dot(y, x, offsetx=ind + 1, offsety=ind + 1, n=m - 1)
        x[ind] = cc * y[ind] - dd
        blas.scal(aa / y[ind], x, n=m - 1, offset=ind + 1)
        blas.axpy(y, x, alpha=dd / y[ind] - cc, n=m - 1, offsetx=ind + 1, offsety=ind + 1)
        blas.scal(1.0 / aa, x, n=m, offset=ind)
        ind += m
    ind2 = ind
    for m in dims['s']:
        for j in range(m):
            u = 0.5 * (y[ind2 + j:ind2 + m] + y[ind2 + j])
            blas.tbsv(u, x, n=m - j, k=0, ldA=1, offsetx=ind + j * (m + 1))
        ind += m * m
        ind2 += m
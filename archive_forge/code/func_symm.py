import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def symm(x, n, offset=0):
    """
    Converts lower triangular matrix to symmetric.  
    Fills in the upper triangular part of the symmetric matrix stored in 
    x[offset : offset+n*n] using 'L' storage.
    """
    if n <= 1:
        return
    for i in range(n - 1):
        blas.copy(x, x, offsetx=offset + i * (n + 1) + 1, offsety=offset + (i + 1) * (n + 1) - 1, incy=n, n=n - i - 1)
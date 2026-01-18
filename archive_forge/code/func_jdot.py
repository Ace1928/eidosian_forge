import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def jdot(x, y, n=None, offsetx=0, offsety=0):
    """
    Returns x' * J * y, where J = [1, 0; 0, -I].
    """
    if n is None:
        if len(x) != len(y):
            raise ValueError('x and y must have the same length')
        n = len(x)
    return x[offsetx] * y[offsety] - blas.dot(x, y, n=n - 1, offsetx=offsetx + 1, offsety=offsety + 1)
import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def jnrm2(x, n=None, offset=0):
    """
    Returns sqrt(x' * J * x) where J = [1, 0; 0, -I], for a vector
    x in a second order cone. 
    """
    if n is None:
        n = len(x)
    a = blas.nrm2(x, n=n - 1, offset=offset + 1)
    return math.sqrt(x[offset] - a) * math.sqrt(x[offset] + a)
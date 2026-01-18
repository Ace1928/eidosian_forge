import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def snrm2(x, dims, mnl=0):
    """
    Returns the norm of a vector in S
    """
    return math.sqrt(sdot(x, x, dims, mnl))
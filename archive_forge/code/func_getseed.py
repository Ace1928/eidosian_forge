import sys
import cvxopt.base
import sys
from cvxopt import printing
from cvxopt.base import matrix, spmatrix, sparse, spdiag, sqrt, sin, cos, \
from cvxopt import solvers, blas, lapack
def getseed():
    """
    Returns the seed value for the random number generator.
    
    getseed()
    """
    try:
        from cvxopt import gsl
        return gsl.getseed()
    except:
        raise NotImplementedError('getseed() not installed (requires GSL)')
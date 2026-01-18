from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
def inequalities(self):
    """ Returns a list of inequality constraints of the LP."""
    return list(self._inequalities)
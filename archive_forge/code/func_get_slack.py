import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def get_slack(self, z):
    return z[self.n_vars:self.n_vars + self.n_ineq]
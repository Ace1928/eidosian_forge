import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def scaling(self, z):
    """Returns scaling vector.
        Given by:
            scaling = [ones(n_vars), s]
        """
    s = self.get_slack(z)
    diag_elements = np.hstack((np.ones(self.n_vars), s))

    def matvec(vec):
        return diag_elements * vec
    return LinearOperator((self.n_vars + self.n_ineq, self.n_vars + self.n_ineq), matvec)
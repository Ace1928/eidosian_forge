import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def lagrangian_hessian(self, z, v):
    """Returns scaled Lagrangian Hessian"""
    Hx = self.lagrangian_hessian_x(z, v)
    if self.n_ineq > 0:
        S_Hs_S = self.lagrangian_hessian_s(z, v)

    def matvec(vec):
        vec_x = self.get_variables(vec)
        vec_s = self.get_slack(vec)
        if self.n_ineq > 0:
            return np.hstack((Hx.dot(vec_x), S_Hs_S * vec_s))
        else:
            return Hx.dot(vec_x)
    return LinearOperator((self.n_vars + self.n_ineq, self.n_vars + self.n_ineq), matvec)
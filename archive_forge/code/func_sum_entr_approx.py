import numpy as np
import scipy as sp
import cvxpy as cp
from cvxpy import trace
from cvxpy.atoms import von_neumann_entr
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.utilities.linalg import onb_for_orthogonal_complement
@staticmethod
def sum_entr_approx(a: cp.Expression, apx_m: int, apx_k: int):
    n = a.size
    epi_vec = cp.Variable(shape=n)
    b = cp.Constant(np.ones(n))
    con = cp.constraints.RelEntrConeQuad(a, b, epi_vec, apx_m, apx_k)
    objective = cp.Minimize(cp.sum(epi_vec))
    return (objective, con)
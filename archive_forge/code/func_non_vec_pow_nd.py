import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
@staticmethod
def non_vec_pow_nd() -> STH.SolverTestHelper:
    n_buyer = 4
    n_items = 6
    z = cp.Variable(shape=(2,))
    np.random.seed(0)
    V = np.random.rand(n_buyer, n_items)
    X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
    u = cp.sum(cp.multiply(V, X), axis=1)
    alpha1 = np.array([0.4069713, 0.10067042, 0.30507361, 0.18728467])
    alpha2 = np.array([0.13209105, 0.18918836, 0.36087677, 0.31784382])
    cons = [cp.PowConeND(u, z[0], alpha1), cp.PowConeND(u, z[1], alpha2), X >= 0, cp.sum(X, axis=0) <= 1]
    obj = cp.Maximize(z[0] + z[1])
    obj_pair = (obj, 2.415600275720486)
    var_pairs = [(X, None), (u, None), (z, None)]
    cons_pairs = [(con, None) for con in cons]
    return STH.SolverTestHelper(obj_pair, var_pairs, cons_pairs)
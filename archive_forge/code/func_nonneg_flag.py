import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
@staticmethod
def nonneg_flag() -> STH.SolverTestHelper:
    """
        Tests nonneg flag
        Reference values via MOSEK
        Version: 10.0.46
        """
    X = cp.Variable(shape=(4, 4), nonneg=True)
    obj = cp.Minimize(cp.log_sum_exp(X))
    cons = [cp.norm2(X) <= 10, X[0, 1] >= 4, X[0, 1] <= 8]
    con_pairs = [(cons[0], None), (cons[1], None), (cons[2], None)]
    var_pairs = [(X, np.array([[1.19672119e-07, 4.0, 1.19672119e-07, 1.19672119e-07], [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309115e-08], [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309115e-08], [8.81309115e-08, 1.19672119e-07, 8.81309115e-08, 8.81309088e-08]]))]
    obj_pair = (obj, 4.242738008082711)
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
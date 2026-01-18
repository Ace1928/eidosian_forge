import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
@staticmethod
def symmetric_flag() -> STH.SolverTestHelper:
    """
        Tests symmetric flag
        Reference values via MOSEK
        Version: 10.0.46
        """
    X = cp.Variable(shape=(4, 4), symmetric=True)
    obj = cp.Minimize(cp.log_sum_exp(X))
    cons = [cp.norm2(X) <= 10, X[0, 1] >= 4, X[0, 1] <= 8]
    con_pairs = [(cons[0], None), (cons[1], None), (cons[2], None)]
    var_pairs = [(X, np.array([[-3.74578525, 4.0, -3.30586268, -3.30586268], [4.0, -3.74578525, -3.30586268, -3.30586268], [-3.30586268, -3.30586268, -2.8684253, -2.8684253], [-3.30586268, -3.30586268, -2.8684253, -2.86842529]]))]
    obj_pair = (obj, 4.698332858812026)
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
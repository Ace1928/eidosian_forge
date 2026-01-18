import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
@staticmethod
def nsd_flag() -> STH.SolverTestHelper:
    """
        Tests NSD flag
        Reference values via MOSEK
        Version: 10.0.46
        """
    X = cp.Variable(shape=(3, 3), NSD=True)
    obj = cp.Maximize(cp.lambda_min(X))
    cons = [X[0, 1] == 123]
    con_pairs = [(cons[0], None)]
    var_pairs = [(X, np.array([[-123.0, 123.0, 0.0], [123.0, -123.0, 0.0], [0.0, 0.0, -123.0]]))]
    obj_pair = (obj, -246.0000000000658)
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
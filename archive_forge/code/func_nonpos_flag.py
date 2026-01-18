import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
@staticmethod
def nonpos_flag() -> STH.SolverTestHelper:
    """
        Tests nonpos flag
        Reference values via MOSEK
        Version: 10.0.46
        """
    X = cp.Variable(shape=(3, 3), nonpos=True)
    obj = cp.Minimize(cp.norm2(X))
    cons = [cp.log_sum_exp(X) <= 2, cp.sum_smallest(X, 5) >= -10]
    con_pairs = [(cons[0], None), (cons[1], None)]
    var_pairs = [(X, np.array([[-0.19722458, -0.19722458, -0.19722457], [-0.19722458, -0.19722458, -0.19722457], [-0.19722457, -0.19722457, -0.19722459]]))]
    obj_pair = (obj, 0.5916737242761841)
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
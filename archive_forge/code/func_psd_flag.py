import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
@staticmethod
def psd_flag() -> STH.SolverTestHelper:
    """
        Tests PSD flag
        Reference values via MOSEK
        Version: 10.0.46
        """
    X = cp.Variable(shape=(4, 4), PSD=True)
    obj = cp.Minimize(cp.log_sum_exp(X))
    cons = [cp.norm2(X) <= 10, X[0, 1] >= 4, X[0, 1] <= 8]
    con_pairs = [(cons[0], None), (cons[1], None), (cons[2], None)]
    var_pairs = [(X, np.array([[4.00000001, 4.0, -1.05467058, -1.05467058], [4.0, 4.00000001, -1.05467058, -1.05467058], [-1.05467058, -1.05467058, 0.27941584, 0.27674984], [-1.05467058, -1.05467058, 0.27674984, 0.27941584]]))]
    obj_pair = (obj, 5.422574709567284)
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
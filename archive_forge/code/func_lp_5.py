import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def lp_5() -> SolverTestHelper:
    x0 = np.array([0, 1, 0, 2, 0, 4, 0, 5, 6, 7])
    mu0 = np.array([-2, -1, 0, 1, 2, 3.5])
    np.random.seed(0)
    A_min = np.random.randn(4, 10)
    A_red = A_min.T @ np.random.rand(4, 2)
    A_red = A_red.T
    A = np.vstack((A_min, A_red))
    b = A @ x0
    c = A.T @ mu0
    c[[0, 2, 4, 6]] += np.random.rand(4)
    x = cp.Variable(10)
    objective = (cp.Minimize(c @ x), c @ x0)
    var_pairs = [(x, x0)]
    con_pairs = [(x >= 0, None), (A @ x == b, None)]
    sth = SolverTestHelper(objective, var_pairs, con_pairs)
    return sth
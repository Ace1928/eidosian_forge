import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def socp_0() -> SolverTestHelper:
    x = cp.Variable(shape=(2,))
    obj_pair = (cp.Minimize(cp.norm(x, 2) + 1), 1)
    con_pairs = [(x == 0, None)]
    var_pairs = [(x, np.array([0, 0]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
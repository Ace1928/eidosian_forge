import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def lp_0() -> SolverTestHelper:
    x = cp.Variable(shape=(2,))
    con_pairs = [(x == 0, None)]
    obj_pair = (cp.Minimize(cp.norm(x, 1) + 1.0), 1)
    var_pairs = [(x, np.array([0, 0]))]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
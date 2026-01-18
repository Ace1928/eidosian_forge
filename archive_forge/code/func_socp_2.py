import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def socp_2() -> SolverTestHelper:
    """
    An (unnecessarily) SOCP-based reformulation of LP_1.
    """
    x = cp.Variable(shape=(2,), name='x')
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    expr = cp.reshape(x[0] + 2 * x[1], (1, 1))
    constraints = [2 * x[0] + x[1] <= 3, cp.constraints.SOC(cp.Constant([3]), expr), x[0] >= 0, x[1] >= 0]
    con_pairs = [(constraints[0], 1), (constraints[1], [np.array([2.0]), np.array([[-2.0]])]), (constraints[2], 0), (constraints[3], 0)]
    var_pairs = [(x, np.array([1, 1]))]
    obj_pair = (objective, -9)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
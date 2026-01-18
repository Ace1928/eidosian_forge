import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_socp_1() -> SolverTestHelper:
    """
    Formulate the following mixed-integer SOCP with cvxpy
        min 3 * x[0] + 2 * x[1] + x[2] +  y[0] + 2 * y[1]
        s.t. norm(x,2) <= y[0]
             norm(x,2) <= y[1]
             x[0] + x[1] + 3*x[2] >= 0.1
             y <= 5, y integer.
    """
    x = cp.Variable(shape=(3,))
    y = cp.Variable(shape=(2,), integer=True)
    constraints = [cp.norm(x, 2) <= y[0], cp.norm(x, 2) <= y[1], x[0] + x[1] + 3 * x[2] >= 0.1, y <= 5]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2] + y[0] + 2 * y[1])
    obj_pair = (obj, 0.21363997604807272)
    var_pairs = [(x, np.array([-0.78510265, -0.43565177, 0.44025147])), (y, np.array([1, 1]))]
    con_pairs = [(c, None) for c in constraints]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
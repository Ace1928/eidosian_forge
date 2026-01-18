import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_lp_1() -> SolverTestHelper:
    x = cp.Variable(2, name='x')
    boolvar = cp.Variable(boolean=True)
    intvar = cp.Variable(integer=True)
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    constraints = [2 * x[0] + x[1] <= intvar, x[0] + 2 * x[1] <= 3 * boolvar, x >= 0, intvar == 3 * boolvar, intvar == 3]
    obj_pair = (objective, -9)
    var_pairs = [(x, np.array([1, 1])), (boolvar, 1), (intvar, 3)]
    con_pairs = [(c, None) for c in constraints]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
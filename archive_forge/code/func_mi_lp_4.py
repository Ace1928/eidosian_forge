import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_lp_4() -> SolverTestHelper:
    """Test MI without constraints"""
    x = cp.Variable(boolean=True)
    from cvxpy.expressions.constants import Constant
    objective = cp.Maximize(Constant(0.23) * x)
    obj_pair = (objective, 0.23)
    var_pairs = [(x, 1)]
    sth = SolverTestHelper(obj_pair, var_pairs, [])
    return sth
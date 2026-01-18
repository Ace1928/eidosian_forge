import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def lp_7() -> SolverTestHelper:
    """
    An ill-posed problem to test multiprecision ability of solvers.

    This test will not pass on CVXOPT (as of v1.3.1) and on SDPA without GMP support.
    """
    n = 50
    a = cp.Variable(n + 1)
    delta = cp.Variable(n)
    b = cp.Variable(n + 1)
    objective = cp.Minimize(cp.sum(cp.pos(delta)))
    constraints = [a[1:] - a[:-1] == delta, a >= cp.pos(b)]
    con_pairs = [(constraints[0], None), (constraints[1], None)]
    var_pairs = [(a, None), (delta, None), (b, None)]
    obj_pair = (objective, 0.0)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
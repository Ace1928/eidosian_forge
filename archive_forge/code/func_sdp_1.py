import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def sdp_1(objective_sense) -> SolverTestHelper:
    """
    Solve "Example 8.3" from Convex Optimization by Boyd & Vandenberghe.

    Verify (1) optimal objective values, (2) that the dual variable to the PSD constraint
    belongs to the correct cone (i.e. the dual variable is itself PSD), and (3) that
    complementary slackness holds with the PSD primal variable and its dual variable.
    """
    rho = cp.Variable(shape=(4, 4), symmetric=True)
    constraints = [0.6 <= rho[0, 1], rho[0, 1] <= 0.9, 0.8 <= rho[0, 2], rho[0, 2] <= 0.9, 0.5 <= rho[1, 3], rho[1, 3] <= 0.7, -0.8 <= rho[2, 3], rho[2, 3] <= -0.4, rho[0, 0] == 1, rho[1, 1] == 1, rho[2, 2] == 1, rho[3, 3] == 1, rho >> 0]
    if objective_sense == 'min':
        obj = cp.Minimize(rho[0, 3])
        obj_pair = (obj, -0.39)
    elif objective_sense == 'max':
        obj = cp.Maximize(rho[0, 3])
        obj_pair = (obj, 0.23)
    else:
        raise RuntimeError('Unknown objective_sense.')
    con_pairs = [(c, None) for c in constraints]
    var_pairs = [(rho, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
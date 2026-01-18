import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_lp_5() -> SolverTestHelper:
    z = cp.Variable(11, boolean=True)
    constraints = [z[2] + z[1] == 1, z[4] + z[3] == 1, z[6] + z[5] == 1, z[8] + z[7] == 1, z[10] + z[9] == 1, z[4] + z[1] <= 1, z[2] + z[3] <= 1, z[6] + z[2] <= 1, z[1] + z[5] <= 1, z[8] + z[6] <= 1, z[5] + z[7] <= 1, z[10] + z[8] <= 1, z[7] + z[9] <= 1, z[9] + z[4] <= 1, z[3] + z[10] <= 1]
    obj = cp.Minimize(0)
    obj_pair = (obj, np.inf)
    con_pairs = [(c, None) for c in constraints]
    var_pairs = [(z, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
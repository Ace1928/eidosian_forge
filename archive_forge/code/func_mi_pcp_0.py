import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_pcp_0() -> SolverTestHelper:
    """
    max  x3 + x4 - x0
    s.t. x0 + x1 + x2 / 2 == 2,
         (x0, x1, x3) in Pow3D(0.2)
         (x2, q, x4) in Pow3D(0.4)
         0.1 <= q <= 1.9,
         q integer
    """
    x = cp.Variable(shape=(3,))
    hypos = cp.Variable(shape=(2,))
    q = cp.Variable(integer=True)
    objective = cp.Minimize(-cp.sum(hypos) + x[0])
    arg1 = cp.hstack([x[0], x[2]])
    arg2 = cp.hstack([x[1], q])
    pc_con = cp.constraints.PowCone3D(arg1, arg2, hypos, [0.2, 0.4])
    con_pairs = [(x[0] + x[1] + 0.5 * x[2] == 2, None), (pc_con, None), (0.1 <= q, None), (q <= 1.9, None)]
    obj_pair = (objective, -1.8073406786220672)
    var_pairs = [(x, np.array([0.06393515, 0.78320961, 2.30571048])), (hypos, None), (q, 1.0)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def sdp_2() -> SolverTestHelper:
    """
    Example SDO2 from MOSEK 9.2 documentation.
    """
    X1 = cp.Variable(shape=(2, 2), symmetric=True)
    X2 = cp.Variable(shape=(4, 4), symmetric=True)
    C1 = np.array([[1, 0], [0, 6]])
    A1 = np.array([[1, 1], [1, 2]])
    C2 = np.array([[1, -3, 0, 0], [-3, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    A2 = np.array([[0, 1, 0, 0], [1, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, -3]])
    b = 23
    k = -3
    var_pairs = [(X1, np.array([[21.04711571, 4.07709873], [4.07709873, 0.7897868]])), (X2, np.array([[5.05366214, -3.0, 0.0, 0.0], [-3.0, 1.78088676, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.0]]))]
    con_pairs = [(cp.trace(A1 @ X1) + cp.trace(A2 @ X2) == b, -0.83772234), (X2[0, 1] <= k, 11.04455278), (X1 >> 0, np.array([[21.04711571, 4.07709873], [4.07709873, 0.7897868]])), (X2 >> 0, np.array([[1.0, 1.68455405, 0.0, 0.0], [1.68455405, 2.83772234, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.51316702]]))]
    obj_expr = cp.Minimize(cp.trace(C1 @ X1) + cp.trace(C2 @ X2))
    obj_pair = (obj_expr, 52.40127214)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
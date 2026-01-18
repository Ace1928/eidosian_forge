import unittest
import numpy as np
import pytest
import scipy as sp
import cvxpy as cp
from cvxpy import settings as s
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as INSTALLED_MI
from cvxpy.reductions.solvers.defines import MI_SOCP_SOLVERS as MI_SOCP
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
@staticmethod
def pcp_3(axis):
    """
        A modification of pcp_2. Reformulate

            max  (x**0.2)*(y**0.8) + z**0.4 - x
            s.t. x + y + z/2 == 2
                 x, y, z >= 0
        Into

            max  x3 + x4 - x0
            s.t. x0 + x1 + x2 / 2 == 2,

                 W := [[x0, x2],
                      [x1, 1.0]]
                 z := [x3, x4]
                 alpha := [[0.2, 0.4],
                          [0.8, 0.6]]
                 (W, z) in PowND(alpha, axis=0)
        """
    x = cp.Variable(shape=(3,))
    expect_x = np.array([0.06393515, 0.78320961, 2.30571048])
    hypos = cp.Variable(shape=(2,))
    expect_hypos = None
    objective = cp.Maximize(cp.sum(hypos) - x[0])
    W = cp.bmat([[x[0], x[2]], [x[1], 1.0]])
    alpha = np.array([[0.2, 0.4], [0.8, 0.6]])
    if axis == 1:
        W = W.T
        alpha = alpha.T
    con_pairs = [(x[0] + x[1] + 0.5 * x[2] == 2, None), (cp.constraints.PowConeND(W, hypos, alpha, axis=axis), None)]
    obj_pair = (objective, 1.8073406786220672)
    var_pairs = [(x, expect_x), (hypos, expect_hypos)]
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth
import numpy as np
import scipy.sparse as sp
from scipy.linalg import lstsq
import cvxpy as cp
from cvxpy import Maximize, Minimize, Parameter, Problem
from cvxpy.atoms import (
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import StandardTestLPs
def square_affine(self, solver) -> None:
    A = np.random.randn(10, 2)
    b = np.random.randn(10)
    p = Problem(Minimize(sum_squares(A @ self.x - b)))
    self.solve_QP(p, solver)
    for var in p.variables():
        self.assertItemsAlmostEqual(lstsq(A, b)[0].flatten(), var.value, places=1)
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
def quad_form_coeff(self, solver) -> None:
    np.random.seed(0)
    A = np.random.randn(5, 5)
    z = np.random.randn(5)
    P = A.T.dot(A)
    q = -2 * P.dot(z)
    p = Problem(Minimize(QuadForm(self.w, P) + q.T @ self.w))
    self.solve_QP(p, solver)
    for var in p.variables():
        self.assertItemsAlmostEqual(z, var.value, places=4)
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
def sparse_system(self, solver) -> None:
    m = 100
    n = 80
    np.random.seed(1)
    density = 0.4
    A = sp.rand(m, n, density)
    b = np.random.randn(m)
    p = Problem(Minimize(sum_squares(A @ self.xs - b)), [self.xs == 0])
    self.solve_QP(p, solver)
    self.assertAlmostEqual(b.T.dot(b), p.value, places=4)
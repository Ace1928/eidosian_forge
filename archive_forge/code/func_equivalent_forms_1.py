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
def equivalent_forms_1(self, solver) -> None:
    m = 100
    n = 80
    r = 70
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    G = np.random.randn(r, n)
    h = np.random.randn(r)
    obj1 = 0.1 * sum((A @ self.xef - b) ** 2)
    cons = [G @ self.xef == h]
    p1 = Problem(Minimize(obj1), cons)
    self.solve_QP(p1, solver)
    self.assertAlmostEqual(p1.value, 68.1119420108, places=4)
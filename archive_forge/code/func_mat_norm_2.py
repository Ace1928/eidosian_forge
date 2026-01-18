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
def mat_norm_2(self, solver) -> None:
    A = np.random.randn(5, 3)
    B = np.random.randn(5, 2)
    p = Problem(Minimize(norm(A @ self.C - B, 2)))
    s = self.solve_QP(p, solver)
    for var in p.variables():
        self.assertItemsAlmostEqual(lstsq(A, B)[0], s.primal_vars[var.id], places=1)
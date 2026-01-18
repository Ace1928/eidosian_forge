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
def quad_over_lin(self, solver) -> None:
    p = Problem(Minimize(0.5 * quad_over_lin(abs(self.x - 1), 1)), [self.x <= -1])
    self.solve_QP(p, solver)
    for var in p.variables():
        self.assertItemsAlmostEqual(np.array([-1.0, -1.0]), var.value, places=4)
    for con in p.constraints:
        self.assertItemsAlmostEqual(np.array([2.0, 2.0]), con.dual_value, places=4)
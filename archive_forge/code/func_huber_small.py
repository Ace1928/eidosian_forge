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
def huber_small(self, solver) -> None:
    x = Variable(3)
    objective = sum(huber(x))
    p = Problem(Minimize(objective), [x[2] >= 3])
    self.solve_QP(p, solver)
    self.assertAlmostEqual(3, x.value[2], places=4)
    self.assertAlmostEqual(5, objective.value, places=4)
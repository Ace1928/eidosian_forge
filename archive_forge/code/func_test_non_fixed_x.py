import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_non_fixed_x(self):
    r = np.array([2, 1, 0, -1, -1])
    w = np.array([1.2, 1.1])
    expr = cp.dotsort(self.x, w)
    prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
    prob.solve()
    self.assertAlmostEqual(expr.value, 1)
    self.assertAlmostEqual(self.x.value[:2] @ w, 1)
    r = np.array([2, 1, 0, -1, -1])
    w = np.array([1.2, 1.1, 1.3])
    expr = cp.dotsort(self.x, w)
    prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
    prob.solve()
    self.assertAlmostEqual(expr.value, 1)
    self.assertAlmostEqual(np.sort(self.x.value)[-3:] @ np.sort(w), 1)
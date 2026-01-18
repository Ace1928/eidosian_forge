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
def test_0D(self):
    x_val = np.array([1, 3, 2, -5, 0])
    w = 1
    expr = cp.dotsort(self.x, w)
    assert expr.is_convex()
    assert expr.is_incr(0)
    prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
    prob.solve()
    self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-1:]))
    x = cp.Variable()
    x_val = np.array([1])
    w = 1
    expr = cp.dotsort(x, w)
    assert expr.is_convex()
    assert expr.is_incr(0)
    prob = cp.Problem(cp.Minimize(expr), [x == x_val])
    prob.solve()
    self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-1:]))
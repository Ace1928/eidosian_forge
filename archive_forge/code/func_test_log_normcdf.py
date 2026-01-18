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
def test_log_normcdf(self) -> None:
    self.assertEqual(cp.log_normcdf(self.x).sign, s.NONPOS)
    self.assertEqual(cp.log_normcdf(self.x).curvature, s.CONCAVE)
    for x in range(-4, 5):
        self.assertAlmostEqual(np.log(scipy.stats.norm.cdf(x)), cp.log_normcdf(x).value, places=None, delta=0.01)
    y = Variable((2, 2))
    obj = Minimize(cp.sum(-cp.log_normcdf(y)))
    prob = Problem(obj, [y == 2])
    result = prob.solve(solver=cp.ECOS)
    self.assertAlmostEqual(-result, 4 * np.log(scipy.stats.norm.cdf(2)), places=None, delta=0.01)
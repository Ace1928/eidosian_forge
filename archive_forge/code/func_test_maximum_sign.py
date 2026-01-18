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
def test_maximum_sign(self) -> None:
    self.assertEqual(cp.maximum(1, 2).sign, s.NONNEG)
    self.assertEqual(cp.maximum(1, Variable()).sign, s.NONNEG)
    self.assertEqual(cp.maximum(1, -2).sign, s.NONNEG)
    self.assertEqual(cp.maximum(1, 0).sign, s.NONNEG)
    self.assertEqual(cp.maximum(Variable(), 0).sign, s.NONNEG)
    self.assertEqual(cp.maximum(Variable(), Variable()).sign, s.UNKNOWN)
    self.assertEqual(cp.maximum(Variable(), -2).sign, s.UNKNOWN)
    self.assertEqual(cp.maximum(0, 0).sign, s.ZERO)
    self.assertEqual(cp.maximum(0, -2).sign, s.ZERO)
    self.assertEqual(cp.maximum(-3, -2).sign, s.NONPOS)
    self.assertEqual(cp.maximum(-2, Variable(), 0, -1, Variable(), 1).sign, s.NONNEG)
    self.assertEqual(cp.maximum(1, Variable(2)).sign, s.NONNEG)
    self.assertEqual(cp.maximum(1, Variable(2)).shape, (2,))
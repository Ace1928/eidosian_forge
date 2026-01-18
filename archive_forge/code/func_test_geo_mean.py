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
def test_geo_mean(self) -> None:
    atom = cp.geo_mean(self.x)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONCAVE)
    self.assertEqual(atom.sign, s.NONNEG)
    copy = atom.copy()
    self.assertTrue(type(copy) is type(atom))
    self.assertEqual(copy.args, atom.args)
    self.assertFalse(copy.args is atom.args)
    self.assertEqual(copy.get_data(), atom.get_data())
    copy = atom.copy(args=[self.y])
    self.assertTrue(type(copy) is type(atom))
    self.assertTrue(copy.args[0] is self.y)
    self.assertEqual(copy.get_data(), atom.get_data())
    with pytest.raises(TypeError, match=SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE):
        cp.geo_mean(self.x, self.y)
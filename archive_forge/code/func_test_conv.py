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
def test_conv(self) -> None:
    """Test the conv atom.
        """
    a = np.ones((3, 1))
    b = Parameter(2, nonneg=True)
    expr = cp.conv(a, b)
    assert expr.is_nonneg()
    self.assertEqual(expr.shape, (4, 1))
    b = Parameter(2, nonpos=True)
    expr = cp.conv(a, b)
    assert expr.is_nonpos()
    with self.assertRaises(Exception) as cm:
        cp.conv(self.x, -1)
    self.assertEqual(str(cm.exception), 'The first argument to conv must be constant.')
    with self.assertRaises(Exception) as cm:
        cp.conv([[0, 1], [0, 1]], self.x)
    self.assertEqual(str(cm.exception), 'The arguments to conv must resolve to vectors.')
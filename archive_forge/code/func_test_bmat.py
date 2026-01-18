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
def test_bmat(self) -> None:
    """Test the bmat atom.
        """
    v_np = np.ones((3, 1))
    expr = np.vstack([np.hstack([v_np, v_np]), np.hstack([np.zeros((2, 1)), np.array([[1, 2]]).T])])
    self.assertEqual(expr.shape, (5, 2))
    const = np.vstack([np.hstack([v_np, v_np]), np.hstack([np.zeros((2, 1)), np.array([[1, 2]]).T])])
    self.assertItemsAlmostEqual(expr, const)
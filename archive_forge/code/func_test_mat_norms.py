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
def test_mat_norms(self) -> None:
    """Test that norm1 and normInf match definition for matrices.
        """
    A = np.array([[1, 2], [3, 4]])
    print(A)
    X = Variable((2, 2))
    obj = Minimize(cp.norm(X, 1))
    prob = cp.Problem(obj, [X == A])
    result = prob.solve(solver=cp.SCS)
    print(result)
    self.assertAlmostEqual(result, cp.norm(A, 1).value, places=3)
    obj = Minimize(cp.norm(X, np.inf))
    prob = cp.Problem(obj, [X == A])
    result = prob.solve(solver=cp.SCS)
    print(result)
    self.assertAlmostEqual(result, cp.norm(A, np.inf).value, places=3)
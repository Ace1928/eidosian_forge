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
def test_mixed_norm(self) -> None:
    """Test mixed norm.
        """
    y = Variable((5, 5))
    obj = Minimize(cp.mixed_norm(y, 'inf', 1))
    prob = Problem(obj, [y == np.ones((5, 5))])
    result = prob.solve(solver=cp.SCS)
    self.assertAlmostEqual(result, 5)
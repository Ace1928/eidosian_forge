import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_missing_imag(self) -> None:
    """Test problems where imaginary is missing.
        """
    Z = Variable((2, 2), hermitian=True)
    constraints = [cp.trace(cp.real(Z)) == 1]
    obj = cp.Minimize(0)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='SCS')
    Z = Variable((2, 2), imag=True)
    obj = cp.Minimize(cp.trace(cp.real(Z)))
    prob = cp.Problem(obj, constraints)
    result = prob.solve(solver='SCS')
    self.assertAlmostEqual(result, 0)
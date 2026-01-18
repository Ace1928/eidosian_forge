import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_multiply_param_and_nonlinear_variable_is_dpp(self) -> None:
    x = cp.Parameter(nonneg=True)
    y = cp.Variable()
    product = x * cp.exp(y)
    self.assertTrue(product.is_convex())
    self.assertTrue(product.is_dcp())
    self.assertTrue(product.is_dpp())
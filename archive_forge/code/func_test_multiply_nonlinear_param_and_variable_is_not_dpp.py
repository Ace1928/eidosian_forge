import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_multiply_nonlinear_param_and_variable_is_not_dpp(self) -> None:
    x = cp.Parameter()
    y = cp.Variable()
    product = cp.exp(x) * y
    self.assertFalse(product.is_dpp())
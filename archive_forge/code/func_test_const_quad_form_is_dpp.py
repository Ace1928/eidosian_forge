import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_const_quad_form_is_dpp(self) -> None:
    x = cp.Variable((2, 1))
    P = np.eye(2)
    y = cp.quad_form(x, P)
    self.assertTrue(y.is_dpp())
    self.assertTrue(y.is_dcp())
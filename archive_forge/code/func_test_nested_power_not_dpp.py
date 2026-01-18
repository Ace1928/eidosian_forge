import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_nested_power_not_dpp(self) -> None:
    alpha = cp.Parameter(value=1.0)
    x = cp.Variable(pos=True)
    pow1 = x ** alpha
    self.assertTrue(pow1.is_dgp(dpp=True))
    pow2 = pow1 ** alpha
    self.assertFalse(pow2.is_dgp(dpp=True))
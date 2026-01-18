import warnings
import numpy as np
import pytest
import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest
def test_mixed_posynomial_is_dpp(self) -> None:
    alpha = cp.Parameter(pos=True)
    beta = cp.Variable(pos=True)
    kappa = cp.Parameter(pos=True)
    tau = cp.Variable(pos=True)
    monomial = alpha ** 1.2 * beta ** 0.5 * kappa ** 3 * kappa ** 2 * tau
    posynomial = (monomial + monomial) ** 3
    self.assertTrue(posynomial.is_dgp(dpp=True))
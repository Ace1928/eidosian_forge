import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_minimize_tnc45(self):
    x0, bnds = ([2] * 5, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
    xopt = [1, 2, 3, 4, 5]
    x = optimize.minimize(self.f45, x0, method='TNC', jac=self.g45, bounds=bnds, options=self.opts).x
    assert_allclose(self.f45(x), self.f45(xopt), atol=1e-08)
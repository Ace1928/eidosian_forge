import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_minimize_tnc3(self):
    x0, bnds = ([10, 1], ([-np.inf, None], [0.0, None]))
    xopt = [0, 0]
    x = optimize.minimize(self.f3, x0, method='TNC', jac=self.g3, bounds=bnds, options=self.opts).x
    assert_allclose(self.f3(x), self.f3(xopt), atol=1e-08)
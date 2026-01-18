import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_minimize_tnc38(self):
    x0, bnds = (np.array([-3, -1, -3, -1]), [(-10, 10)] * 4)
    xopt = [1] * 4
    x = optimize.minimize(self.f38, x0, method='TNC', jac=self.g38, bounds=bnds, options=self.opts).x
    assert_allclose(self.f38(x), self.f38(xopt), atol=1e-08)
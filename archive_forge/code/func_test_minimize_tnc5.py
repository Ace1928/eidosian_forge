import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_minimize_tnc5(self):
    x0, bnds = ([0, 0], [(-1.5, 4), (-3, 3)])
    xopt = [-0.5471975511965976, -1.5471975511965976]
    x = optimize.minimize(self.f5, x0, method='TNC', jac=self.g5, bounds=bnds, options=self.opts).x
    assert_allclose(self.f5(x), self.f5(xopt), atol=1e-08)
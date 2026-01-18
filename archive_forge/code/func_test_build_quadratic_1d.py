from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_build_quadratic_1d(self):
    s = np.zeros(2)
    a, b = build_quadratic_1d(self.J, self.g, s)
    assert_equal(a, 0)
    assert_equal(b, 0)
    a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
    assert_equal(a, 0)
    assert_equal(b, 0)
    s = np.array([1.0, -1.0])
    a, b = build_quadratic_1d(self.J, self.g, s)
    assert_equal(a, 2.05)
    assert_equal(b, 2.8)
    a, b = build_quadratic_1d(self.J, self.g, s, diag=self.diag)
    assert_equal(a, 3.55)
    assert_equal(b, 2.8)
    s0 = np.array([0.5, 0.5])
    a, b, c = build_quadratic_1d(self.J, self.g, s, diag=self.diag, s0=s0)
    assert_equal(a, 3.55)
    assert_allclose(b, 2.39)
    assert_allclose(c, -0.1525)
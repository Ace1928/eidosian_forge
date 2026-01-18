from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_evaluate_quadratic(self):
    s = np.array([1.0, -1.0])
    value = evaluate_quadratic(self.J, self.g, s)
    assert_equal(value, 4.85)
    value = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
    assert_equal(value, 6.35)
    s = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    values = evaluate_quadratic(self.J, self.g, s)
    assert_allclose(values, [4.85, -0.91, 0.0])
    values = evaluate_quadratic(self.J, self.g, s, diag=self.diag)
    assert_allclose(values, [6.35, 0.59, 0.0])
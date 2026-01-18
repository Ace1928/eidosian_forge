import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_exponential(self):
    degree = 5
    p = approximate_taylor_polynomial(np.exp, 0, degree, 1, 15)
    for i in range(degree + 1):
        assert_almost_equal(p(0), 1)
        p = p.deriv()
    assert_almost_equal(p(0), 0)
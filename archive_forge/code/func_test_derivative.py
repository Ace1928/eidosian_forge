import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_derivative(self):
    P = BarycentricInterpolator(self.xs, self.ys)
    m = 10
    r = P.derivatives(self.test_xs, m)
    for i in range(m):
        assert_allclose(P.derivative(self.test_xs, i), r[i])
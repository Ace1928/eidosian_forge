import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_derivatives(self):
    P = BarycentricInterpolator(self.xs, self.ys)
    D = P.derivatives(self.test_xs)
    for i in range(D.shape[0]):
        assert_allclose(self.true_poly.deriv(i)(self.test_xs), D[i])
import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_high_derivative(self):
    P = BarycentricInterpolator(self.xs, self.ys)
    for i in range(len(self.xs), 5 * len(self.xs)):
        assert_allclose(P.derivative(self.test_xs, i), np.zeros(len(self.test_xs)))
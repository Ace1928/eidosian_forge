import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_int_inputs(self):
    x = [0, 234, 468, 702, 936, 1170, 1404, 2340, 3744, 6084, 8424, 13104, 60000]
    offset_cdf = np.array([-0.95, -0.86114777, -0.8147762, -0.64072425, -0.48002351, -0.34925329, -0.26503107, -0.13148093, -0.12988833, -0.12979296, -0.12973574, -0.08582937, 0.05])
    f = KroghInterpolator(x, offset_cdf)
    assert_allclose(abs((f(x) - offset_cdf) / f.derivative(x, 1)), 0, atol=1e-10)
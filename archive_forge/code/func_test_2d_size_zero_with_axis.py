import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
@pytest.mark.parametrize('axis, expected', [(0, []), (1, [np.nan] * 3), (None, np.nan)])
def test_2d_size_zero_with_axis(self, axis, expected):
    x = np.empty((3, 0))
    y = variation(x, axis=axis)
    assert_equal(y, expected)
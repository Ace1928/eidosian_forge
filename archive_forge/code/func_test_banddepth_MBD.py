import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
def test_banddepth_MBD():
    xx = np.arange(5001) / 5000.0
    y1 = np.zeros(xx.shape)
    y2 = 2 * xx - 1
    y3 = np.ones(xx.shape) * 0.5
    y4 = np.ones(xx.shape) * -0.25
    data = np.asarray([y1, y2, y3, y4])
    depth = banddepth(data, method='MBD')
    expected_depth = [5.0 / 6, (2 * (0.75 - 3.0 / 8) + 3) / 6, 3.5 / 6, (2 * 3.0 / 8 + 3) / 6]
    assert_almost_equal(depth, expected_depth, decimal=4)
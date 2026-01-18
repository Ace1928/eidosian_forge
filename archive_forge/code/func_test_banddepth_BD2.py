import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.datasets import elnino
from statsmodels.graphics.functional import (
def test_banddepth_BD2():
    xx = np.arange(500) / 150.0
    y1 = 1 + 0.5 * np.sin(xx)
    y2 = 0.3 + np.sin(xx + np.pi / 6)
    y3 = -0.5 + np.sin(xx + np.pi / 6)
    y4 = -1 + 0.3 * np.cos(xx + np.pi / 6)
    data = np.asarray([y1, y2, y3, y4])
    depth = banddepth(data, method='BD2')
    expected_depth = [0.5, 5.0 / 6, 5.0 / 6, 0.5]
    assert_almost_equal(depth, expected_depth)
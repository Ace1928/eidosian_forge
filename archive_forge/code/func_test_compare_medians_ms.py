import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_compare_medians_ms():
    x = np.arange(7)
    y = x + 10
    assert_almost_equal(ms.compare_medians_ms(x, y), 0)
    y2 = np.linspace(0, 1, num=10)
    assert_almost_equal(ms.compare_medians_ms(x, y2), 0.017116406778)
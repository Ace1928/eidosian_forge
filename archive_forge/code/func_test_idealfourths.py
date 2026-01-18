import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_idealfourths():
    test = np.arange(100)
    assert_almost_equal(np.asarray(ms.idealfourths(test)), [24.416667, 74.583333], 6)
    test_2D = test.repeat(3).reshape(-1, 3)
    assert_almost_equal(ms.idealfourths(test_2D, axis=0), [[24.416667, 24.416667, 24.416667], [74.583333, 74.583333, 74.583333]], 6)
    assert_almost_equal(ms.idealfourths(test_2D, axis=1), test.repeat(2).reshape(-1, 2))
    test = [0, 0]
    _result = ms.idealfourths(test)
    assert_(np.isnan(_result).all())
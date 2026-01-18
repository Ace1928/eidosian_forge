import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_median_cihs():
    rng = np.random.default_rng(8824288259505800535)
    x = rng.random(size=20)
    assert_allclose(ms.median_cihs(x), (0.38663198, 0.88431272))
    assert_allclose(ms.median_cihs(x, 0.1), (0.48319773366, 0.8809426805))
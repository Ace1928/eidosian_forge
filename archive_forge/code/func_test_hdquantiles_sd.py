import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_hdquantiles_sd(self):
    hd_std_errs = ms.hdquantiles_sd(self.data)
    n = len(self.data)
    jdata = np.broadcast_to(self.data, (n, n))
    jselector = np.logical_not(np.eye(n))
    jdata = jdata[jselector].reshape(n, n - 1)
    jdist = ms.hdquantiles(jdata, axis=1)
    jdist_mean = np.mean(jdist, axis=0)
    jstd = ((n - 1) / n * np.sum((jdist - jdist_mean) ** 2, axis=0)) ** 0.5
    assert_almost_equal(hd_std_errs, jstd)
    assert_almost_equal(hd_std_errs, [0.0379258, 0.0380656, 0.0380013])
    two_data_points = ms.hdquantiles_sd([1, 2])
    assert_almost_equal(two_data_points, [0.5, 0.5, 0.5])
import numpy as np
import numpy.ma as ma
import scipy.stats.mstats as ms
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
def test_hdquantiles(self):
    data = self.data
    assert_almost_equal(ms.hdquantiles(data, [0.0, 1.0]), [0.006514031, 0.995309248])
    hdq = ms.hdquantiles(data, [0.25, 0.5, 0.75])
    assert_almost_equal(hdq, [0.253210762, 0.512847491, 0.762232442])
    data = np.array(data).reshape(10, 10)
    hdq = ms.hdquantiles(data, [0.25, 0.5, 0.75], axis=0)
    assert_almost_equal(hdq[:, 0], ms.hdquantiles(data[:, 0], [0.25, 0.5, 0.75]))
    assert_almost_equal(hdq[:, -1], ms.hdquantiles(data[:, -1], [0.25, 0.5, 0.75]))
    hdq = ms.hdquantiles(data, [0.25, 0.5, 0.75], axis=0, var=True)
    assert_almost_equal(hdq[..., 0], ms.hdquantiles(data[:, 0], [0.25, 0.5, 0.75], var=True))
    assert_almost_equal(hdq[..., -1], ms.hdquantiles(data[:, -1], [0.25, 0.5, 0.75], var=True))
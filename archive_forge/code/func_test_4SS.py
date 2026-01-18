from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.motion
import skvideo.datasets
def test_4SS():
    dat = getmockdata()
    mvec = skvideo.motion.blockMotion(dat, '4SS')
    mvec = mvec.astype(np.float32)
    xmean = np.mean(mvec[:, :, :, 0])
    ymean = np.mean(mvec[:, :, :, 1])
    assert_almost_equal(ymean, -0.770833313465118, decimal=15)
    assert_almost_equal(xmean, -0.055555555969477, decimal=15)
from numpy.testing import assert_equal, assert_almost_equal
import os
import sys
import numpy as np
import skvideo.io
import skvideo.motion
import skvideo.datasets
def test_N3SS():
    dat = getmockdata()
    mvec = skvideo.motion.blockMotion(dat, 'N3SS')
    mvec = mvec.astype(np.float32)
    xmean = np.mean(mvec[:, :, :, 0])
    ymean = np.mean(mvec[:, :, :, 1])
    assert_equal(xmean, 0.0)
    assert_equal(ymean, 0)
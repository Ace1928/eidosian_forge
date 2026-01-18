from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_cohere():
    N = 1024
    np.random.seed(19680801)
    x = np.random.randn(N)
    y = np.roll(x, 20)
    y = np.convolve(y, np.ones(20) / 20.0, mode='same')
    cohsq, f = mlab.cohere(x, y, NFFT=256, Fs=2, noverlap=128)
    assert_allclose(np.mean(cohsq), 0.837, atol=0.001)
    assert np.isreal(np.mean(cohsq))
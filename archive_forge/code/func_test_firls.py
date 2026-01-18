import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_firls(self):
    N = 11
    a = 0.1
    h = firls(11, [0, a, 0.5 - a, 0.5], [1, 1, 0, 0], fs=1.0)
    assert_equal(len(h), N)
    midx = (N - 1) // 2
    assert_array_almost_equal(h[:midx], h[:-midx - 1:-1])
    assert_almost_equal(h[midx], 0.5)
    hodd = np.hstack((h[1:midx:2], h[-midx + 1::2]))
    assert_array_almost_equal(hodd, 0)
    w, H = freqz(h, 1)
    f = w / 2 / np.pi
    Hmag = np.abs(H)
    idx = np.logical_and(f > 0, f < a)
    assert_array_almost_equal(Hmag[idx], 1, decimal=3)
    idx = np.logical_and(f > 0.5 - a, f < 0.5)
    assert_array_almost_equal(Hmag[idx], 0, decimal=3)
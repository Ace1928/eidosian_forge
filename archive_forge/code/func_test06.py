import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test06(self):
    """Test firwin2 for calculating Type III filters"""
    ntaps = 1501
    freq = [0.0, 0.5, 0.55, 1.0]
    gain = [0.0, 0.5, 0.0, 0.0]
    taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
    assert_equal(taps[ntaps // 2], 0.0)
    assert_array_almost_equal(taps[:ntaps // 2], -taps[ntaps // 2 + 1:][::-1])
    freqs, response1 = freqz(taps, worN=2048)
    response2 = np.interp(freqs / np.pi, freq, gain)
    assert_array_almost_equal(abs(response1), response2, decimal=3)
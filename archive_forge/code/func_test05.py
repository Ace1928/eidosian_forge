import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test05(self):
    """Test firwin2 for calculating Type IV filters"""
    ntaps = 1500
    freq = [0.0, 1.0]
    gain = [0.0, 1.0]
    taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
    assert_array_almost_equal(taps[:ntaps // 2], -taps[ntaps // 2:][::-1])
    freqs, response = freqz(taps, worN=2048)
    assert_array_almost_equal(abs(response), freqs / np.pi, decimal=4)
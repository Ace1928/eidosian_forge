import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test03(self):
    width = 0.02
    ntaps, beta = kaiserord(120, width)
    ntaps = int(ntaps) | 1
    freq = [0.0, 0.4, 0.4, 0.5, 0.5, 1.0]
    gain = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
    freq_samples = np.array([0.0, 0.4 - width, 0.4 + width, 0.45, 0.5 - width, 0.5 + width, 0.75, 1.0])
    freqs, response = freqz(taps, worN=np.pi * freq_samples)
    assert_array_almost_equal(np.abs(response), [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)
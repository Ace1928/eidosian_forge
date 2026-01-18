import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_bandpass(self):
    width = 0.04
    ntaps, beta = kaiserord(120, width)
    kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
    taps = firwin(ntaps, pass_zero=False, **kwargs)
    assert_array_almost_equal(taps[:ntaps // 2], taps[ntaps:ntaps - ntaps // 2 - 1:-1])
    freq_samples = np.array([0.0, 0.2, 0.3 - width / 2, 0.3 + width / 2, 0.5, 0.7 - width / 2, 0.7 + width / 2, 0.8, 1.0])
    freqs, response = freqz(taps, worN=np.pi * freq_samples)
    assert_array_almost_equal(np.abs(response), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
    taps_str = firwin(ntaps, pass_zero='bandpass', **kwargs)
    assert_allclose(taps, taps_str)
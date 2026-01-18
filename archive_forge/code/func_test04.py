import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test04(self):
    """Test firwin2 when window=None."""
    ntaps = 5
    freq = [0.0, 0.5, 0.5, 1.0]
    gain = [1.0, 1.0, 0.0, 0.0]
    taps = firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
    alpha = 0.5 * (ntaps - 1)
    m = np.arange(0, ntaps) - alpha
    h = 0.5 * sinc(0.5 * m)
    assert_array_almost_equal(h, taps)
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def mse(self, h, bands):
    """Compute mean squared error versus ideal response across frequency
        band.
          h -- coefficients
          bands -- list of (left, right) tuples relative to 1==Nyquist of
            passbands
        """
    w, H = freqz(h, worN=1024)
    f = w / np.pi
    passIndicator = np.zeros(len(w), bool)
    for left, right in bands:
        passIndicator |= (f >= left) & (f < right)
    Hideal = np.where(passIndicator, 1, 0)
    mse = np.mean(abs(abs(H) - Hideal) ** 2)
    return mse
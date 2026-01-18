import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_extremes(self):
    lam = windows.dpss(31, 6, 4, return_ratios=True)[1]
    assert_array_almost_equal(lam, 1.0)
    lam = windows.dpss(31, 7, 4, return_ratios=True)[1]
    assert_array_almost_equal(lam, 1.0)
    lam = windows.dpss(31, 8, 4, return_ratios=True)[1]
    assert_array_almost_equal(lam, 1.0)
import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_cheb_odd_low_attenuation(self):
    cheb_odd_low_at_true = array([1.0, 0.519052, 0.586405, 0.610151, 0.586405, 0.519052, 1.0])
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'This window is not suitable')
        cheb_odd = windows.chebwin(7, at=10)
    assert_array_almost_equal(cheb_odd, cheb_odd_low_at_true, decimal=4)
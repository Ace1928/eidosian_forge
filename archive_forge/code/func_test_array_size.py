import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_array_size(self):
    for n in [0, 10, 11]:
        assert_equal(len(windows.lanczos(n, sym=False)), n)
        assert_equal(len(windows.lanczos(n, sym=True)), n)
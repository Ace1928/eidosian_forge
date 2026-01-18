import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_general_cosine(self):
    assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4), [0.4, 0.3, 1, 0.3])
    assert_allclose(get_window(('general_cosine', [0.5, 0.3, 0.2]), 4, fftbins=False), [0.4, 0.55, 0.55, 0.4])
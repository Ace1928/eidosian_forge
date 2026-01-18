import sys
import numpy as np
from numpy.testing import (assert_, assert_approx_equal,
import pytest
from pytest import raises as assert_raises
from scipy import signal
from scipy.fft import fftfreq
from scipy.integrate import trapezoid
from scipy.signal import (periodogram, welch, lombscargle, coherence,
from scipy.signal._spectral_py import _spectral_helper
from scipy.signal.tests._scipy_spectral_test_shim import stft_compare as stft
from scipy.signal.tests._scipy_spectral_test_shim import istft_compare as istft
from scipy.signal.tests._scipy_spectral_test_shim import csd_compare as csd
def test_empty_input_other_axis(self):
    for shape in [(3, 0), (0, 5, 2)]:
        f, p = csd(np.empty(shape), np.empty(shape), axis=1)
        assert_array_equal(f.shape, shape)
        assert_array_equal(p.shape, shape)
    f, p = csd(np.empty((10, 10, 3)), np.zeros((10, 0, 1)), axis=1)
    assert_array_equal(f.shape, (10, 0, 3))
    assert_array_equal(p.shape, (10, 0, 3))
    f, p = csd(np.empty((10, 0, 1)), np.zeros((10, 10, 3)), axis=1)
    assert_array_equal(f.shape, (10, 0, 3))
    assert_array_equal(p.shape, (10, 0, 3))
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
def test_integer_odd(self):
    x = np.zeros(15, dtype=int)
    x[0] = 1
    f, p = periodogram(x)
    assert_allclose(f, np.arange(8.0) / 15.0)
    q = np.ones(8)
    q[0] = 0
    q *= 2.0 / 15.0
    assert_allclose(p, q, atol=1e-15)
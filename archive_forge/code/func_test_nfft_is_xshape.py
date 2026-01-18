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
def test_nfft_is_xshape(self):
    x = np.zeros(16)
    x[0] = 1
    f, p = periodogram(x, nfft=16)
    assert_allclose(f, np.linspace(0, 0.5, 9))
    q = np.ones(9)
    q[0] = 0
    q[-1] /= 2.0
    q /= 8
    assert_allclose(p, q)
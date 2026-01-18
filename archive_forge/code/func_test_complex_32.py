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
def test_complex_32(self):
    x = np.zeros(16, 'F')
    x[0] = 1.0 + 2j
    x[8] = 1.0 + 2j
    f, p = csd(x, x, nperseg=8, return_onesided=False)
    assert_allclose(f, fftfreq(8, 1.0))
    q = np.array([0.41666666, 0.38194442, 0.55555552, 0.55555552, 0.55555558, 0.55555552, 0.55555552, 0.38194442], 'f')
    assert_allclose(p, q, atol=1e-07, rtol=1e-07)
    assert_(p.dtype == q.dtype, f'dtype mismatch, {p.dtype}, {q.dtype}')
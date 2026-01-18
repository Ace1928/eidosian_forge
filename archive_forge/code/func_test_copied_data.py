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
def test_copied_data(self):
    x = np.random.randn(64)
    y = x.copy()
    _, p_same = csd(x, x, nperseg=8, average='mean', return_onesided=False)
    _, p_copied = csd(x, y, nperseg=8, average='mean', return_onesided=False)
    assert_allclose(p_same, p_copied)
    _, p_same = csd(x, x, nperseg=8, average='median', return_onesided=False)
    _, p_copied = csd(x, y, nperseg=8, average='median', return_onesided=False)
    assert_allclose(p_same, p_copied)
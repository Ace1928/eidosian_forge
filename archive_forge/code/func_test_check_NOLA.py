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
def test_check_NOLA(self):
    settings_pass = [('boxcar', 10, 0), ('boxcar', 10, 9), ('boxcar', 10, 7), ('bartlett', 51, 26), ('bartlett', 51, 10), ('hann', 256, 128), ('hann', 256, 192), ('hann', 256, 37), ('blackman', 300, 200), ('blackman', 300, 123), (('tukey', 0.5), 256, 64), (('tukey', 0.5), 256, 38), ('hann', 256, 255), ('hann', 256, 39)]
    for setting in settings_pass:
        msg = '{}, {}, {}'.format(*setting)
        assert_equal(True, check_NOLA(*setting), err_msg=msg)
    w_fail = np.ones(16)
    w_fail[::2] = 0
    settings_fail = [(w_fail, len(w_fail), len(w_fail) // 2), ('hann', 64, 0)]
    for setting in settings_fail:
        msg = '{}, {}, {}'.format(*setting)
        assert_equal(False, check_NOLA(*setting), err_msg=msg)
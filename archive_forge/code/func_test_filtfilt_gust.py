import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
def test_filtfilt_gust():
    z, p, k = signal.ellip(3, 0.01, 120, 0.0875, output='zpk')
    eps = 1e-10
    r = np.max(np.abs(p))
    approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
    np.random.seed(123)
    b, a = zpk2tf(z, p, k)
    for irlen in [None, approx_impulse_len]:
        signal_len = 5 * approx_impulse_len
        check_filtfilt_gust(b, a, (signal_len,), 0, irlen)
        for axis in range(3):
            shape = [2, 2, 2]
            shape[axis] = signal_len
            check_filtfilt_gust(b, a, shape, axis, irlen)
    length = 2 * approx_impulse_len - 50
    check_filtfilt_gust(b, a, (length,), 0, approx_impulse_len)
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
def test_initial_conditions(self, dt):
    b1, a1 = signal.butter(2, 0.25, 'low')
    b2, a2 = signal.butter(2, 0.75, 'low')
    b3, a3 = signal.butter(2, 0.75, 'low')
    b = np.convolve(np.convolve(b1, b2), b3)
    a = np.convolve(np.convolve(a1, a2), a3)
    sos = np.array((np.r_[b1, a1], np.r_[b2, a2], np.r_[b3, a3]))
    x = np.random.rand(50).astype(dt)
    y_true, zi = lfilter(b, a, x[:20], zi=np.zeros(6))
    y_true = np.r_[y_true, lfilter(b, a, x[20:], zi=zi)[0]]
    assert_allclose_cast(y_true, lfilter(b, a, x))
    y_sos, zi = sosfilt(sos, x[:20], zi=np.zeros((3, 2)))
    y_sos = np.r_[y_sos, sosfilt(sos, x[20:], zi=zi)[0]]
    assert_allclose_cast(y_true, y_sos)
    zi = sosfilt_zi(sos)
    x = np.ones(8, dt)
    y, zf = sosfilt(sos, x, zi=zi)
    assert_allclose_cast(y, np.ones(8))
    assert_allclose_cast(zf, zi)
    x.shape = (1, 1) + x.shape
    assert_raises(ValueError, sosfilt, sos, x, zi=zi)
    zi_nd = zi.copy()
    zi_nd.shape = (zi.shape[0], 1, 1, zi.shape[-1])
    assert_raises(ValueError, sosfilt, sos, x, zi=zi_nd[:, :, :, [0, 1, 1]])
    y, zf = sosfilt(sos, x, zi=zi_nd)
    assert_allclose_cast(y[0, 0], np.ones(8))
    assert_allclose_cast(zf[:, 0, 0, :], zi)
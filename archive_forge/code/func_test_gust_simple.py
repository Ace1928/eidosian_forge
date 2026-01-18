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
def test_gust_simple(self):
    if self.filtfilt_kind != 'tf':
        pytest.skip('gust only implemented for TF systems')
    x = np.array([1.0, 2.0])
    b = np.array([0.5])
    a = np.array([1.0, -0.5])
    y, z1, z2 = _filtfilt_gust(b, a, x)
    assert_allclose([z1[0], z2[0]], [0.3 * x[0] + 0.2 * x[1], 0.2 * x[0] + 0.3 * x[1]])
    assert_allclose(y, [z1[0] + 0.25 * z2[0] + 0.25 * x[0] + 0.125 * x[1], 0.25 * z1[0] + z2[0] + 0.125 * x[0] + 0.25 * x[1]])
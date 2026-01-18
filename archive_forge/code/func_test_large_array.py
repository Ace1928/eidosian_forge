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
@pytest.mark.slow
@pytest.mark.xfail_on_32bit("Can't create large array for test")
def test_large_array(self):
    n = 2 ** 31 // (1000 * np.int64().itemsize)
    _testutils.check_free_memory(2 * n * 1001 * np.int64().itemsize / 1000000.0)
    a = np.zeros(1001 * n, dtype=np.int64)
    a[::2] = 1
    a = np.lib.stride_tricks.as_strided(a, shape=(n, 1000), strides=(8008, 8))
    count = signal.convolve2d(a, [[1, 1]])
    fails = np.where(count > 1)
    assert fails[0].size == 0
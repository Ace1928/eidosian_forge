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
def test_compute_factors(self):
    factors, poly = _compute_factors([1, 2, 3], [3, 2, 1])
    assert_equal(len(factors), 3)
    assert_almost_equal(factors[0], np.poly([2, 2, 3]))
    assert_almost_equal(factors[1], np.poly([1, 1, 1, 3]))
    assert_almost_equal(factors[2], np.poly([1, 1, 1, 2, 2]))
    assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))
    factors, poly = _compute_factors([1, 2, 3], [3, 2, 1], include_powers=True)
    assert_equal(len(factors), 6)
    assert_almost_equal(factors[0], np.poly([1, 1, 2, 2, 3]))
    assert_almost_equal(factors[1], np.poly([1, 2, 2, 3]))
    assert_almost_equal(factors[2], np.poly([2, 2, 3]))
    assert_almost_equal(factors[3], np.poly([1, 1, 1, 2, 3]))
    assert_almost_equal(factors[4], np.poly([1, 1, 1, 3]))
    assert_almost_equal(factors[5], np.poly([1, 1, 1, 2, 2]))
    assert_almost_equal(poly, np.poly([1, 1, 1, 2, 2, 3]))
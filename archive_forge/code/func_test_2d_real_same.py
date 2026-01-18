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
@pytest.mark.parametrize('axes', ['', None, [0, 1], [1, 0], [0, -1], [-1, 0], [-2, 1], [1, -2], [-2, -1], [-1, -2]])
def test_2d_real_same(self, axes):
    a = array([[1, 2, 3], [4, 5, 6]])
    expected = array([[1, 4, 10, 12, 9], [8, 26, 56, 54, 36], [16, 40, 73, 60, 36]])
    if axes == '':
        out = fftconvolve(a, a)
    else:
        out = fftconvolve(a, a, axes=axes)
    assert_array_almost_equal(out, expected)
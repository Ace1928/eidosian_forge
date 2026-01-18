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
def test_2d_complex_same(self, axes):
    a = array([[1 + 2j, 3 + 4j, 5 + 6j], [2 + 1j, 4 + 3j, 6 + 5j]])
    expected = array([[-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j], [10j, 44j, 118j, 156j, 122j], [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j]])
    if axes == '':
        out = fftconvolve(a, a)
    else:
        out = fftconvolve(a, a, axes=axes)
    assert_array_almost_equal(out, expected)
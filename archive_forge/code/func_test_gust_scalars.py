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
def test_gust_scalars(self):
    if self.filtfilt_kind != 'tf':
        pytest.skip('gust only implemented for TF systems')
    x = np.arange(12)
    b = 3.0
    a = 2.0
    y = filtfilt(b, a, x, method='gust')
    expected = (b / a) ** 2 * x
    assert_allclose(y, expected)
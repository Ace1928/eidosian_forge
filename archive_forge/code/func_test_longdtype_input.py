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
@pytest.mark.parametrize('dtype', [np.longdouble, np.clongdouble])
def test_longdtype_input(self, dtype):
    x = np.random.random((27, 27)).astype(dtype)
    y = np.random.random((4, 4)).astype(dtype)
    if np.iscomplexobj(dtype()):
        x += 0.1j
        y -= 0.1j
    res = fftconvolve(x, y)
    assert_allclose(res, convolve(x, y, method='direct'))
    assert res.dtype == dtype
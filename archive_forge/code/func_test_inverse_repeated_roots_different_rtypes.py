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
def test_inverse_repeated_roots_different_rtypes(self):
    r = [3 / 20, -7 / 36, -1 / 6, 2 / 45]
    p = [0, -2, -2, -5]
    k = []
    b_expected = [0, 0, 1, 3]
    b_expected_z = [-1 / 6, -2 / 3, 11 / 6, 3]
    a_expected = [1, 9, 24, 20, 0]
    for rtype in ('avg', 'mean', 'min', 'minimum', 'max', 'maximum'):
        b, a = invres(r, p, k, rtype=rtype)
        assert_allclose(b, b_expected, atol=1e-14)
        assert_allclose(a, a_expected)
        b, a = invresz(r, p, k, rtype=rtype)
        assert_allclose(b, b_expected_z, atol=1e-14)
        assert_allclose(a, a_expected)
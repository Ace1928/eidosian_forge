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
def test_choose_conv_method():
    for mode in ['valid', 'same', 'full']:
        for ndim in [1, 2]:
            n, k, true_method = (8, 6, 'direct')
            x = np.random.randn(*(n,) * ndim)
            h = np.random.randn(*(k,) * ndim)
            method = choose_conv_method(x, h, mode=mode)
            assert_equal(method, true_method)
            method_try, times = choose_conv_method(x, h, mode=mode, measure=True)
            assert_(method_try in {'fft', 'direct'})
            assert_(isinstance(times, dict))
            assert_('fft' in times.keys() and 'direct' in times.keys())
        n = 10
        for not_fft_conv_supp in ['complex256', 'complex192']:
            if hasattr(np, not_fft_conv_supp):
                x = np.ones(n, dtype=not_fft_conv_supp)
                h = x.copy()
                assert_equal(choose_conv_method(x, h, mode=mode), 'direct')
        x = np.array([2 ** 51], dtype=np.int64)
        h = x.copy()
        assert_equal(choose_conv_method(x, h, mode=mode), 'direct')
        x = [Decimal(3), Decimal(2)]
        h = [Decimal(1), Decimal(4)]
        assert_equal(choose_conv_method(x, h, mode=mode), 'direct')
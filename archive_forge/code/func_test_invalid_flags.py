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
@pytest.mark.parametrize('convapproach', [fftconvolve, oaconvolve])
def test_invalid_flags(self, convapproach):
    with assert_raises(ValueError, match="acceptable mode flags are 'valid', 'same', or 'full'"):
        convapproach([1], [2], mode='chips')
    with assert_raises(ValueError, match='when provided, axes cannot be empty'):
        convapproach([1], [2], axes=[])
    with assert_raises(ValueError, match='axes must be a scalar or iterable of integers'):
        convapproach([1], [2], axes=[[1, 2], [3, 4]])
    with assert_raises(ValueError, match='axes must be a scalar or iterable of integers'):
        convapproach([1], [2], axes=[1.0, 2.0, 3.0, 4.0])
    with assert_raises(ValueError, match='axes exceeds dimensionality of input'):
        convapproach([1], [2], axes=[1])
    with assert_raises(ValueError, match='axes exceeds dimensionality of input'):
        convapproach([1], [2], axes=[-2])
    with assert_raises(ValueError, match='all axes must be unique'):
        convapproach([1], [2], axes=[0, 0])
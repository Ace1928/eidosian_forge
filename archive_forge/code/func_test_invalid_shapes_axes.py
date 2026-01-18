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
def test_invalid_shapes_axes(self, convapproach):
    a = np.zeros([5, 6, 2, 1])
    b = np.zeros([5, 6, 3, 1])
    with assert_raises(ValueError, match='incompatible shapes for in1 and in2: \\(5L?, 6L?, 2L?, 1L?\\) and \\(5L?, 6L?, 3L?, 1L?\\)'):
        convapproach(a, b, axes=[0, 1])
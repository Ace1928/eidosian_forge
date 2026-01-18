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
@staticmethod
def assert_rp_almost_equal(r, p, r_true, p_true, decimal=7):
    r_true = np.asarray(r_true)
    p_true = np.asarray(p_true)
    distance = np.hypot(abs(p[:, None] - p_true), abs(r[:, None] - r_true))
    rows, cols = linear_sum_assignment(distance)
    assert_almost_equal(p[rows], p_true[cols], decimal=decimal)
    assert_almost_equal(r[rows], r_true[cols], decimal=decimal)
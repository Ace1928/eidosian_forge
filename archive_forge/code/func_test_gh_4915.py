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
def test_gh_4915(self):
    p = np.roots(np.convolve(np.ones(5), np.ones(5)))
    true_roots = [-(-1) ** (1 / 5), (-1) ** (4 / 5), -(-1) ** (3 / 5), (-1) ** (2 / 5)]
    unique, multiplicity = unique_roots(p)
    unique = np.sort(unique)
    assert_almost_equal(np.sort(unique), true_roots, decimal=7)
    assert_equal(multiplicity, [2, 2, 2, 2])
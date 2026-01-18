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
@pytest.mark.parametrize('bp', [np.array([0, 2]), [0, 2]])
def test_detrend_array_bp(self, bp):
    rng = np.random.RandomState(12345)
    x = rng.rand(10)
    res = detrend(x, bp=bp)
    res_scipy_191 = np.array([-4.4408921e-16, -2.22044605e-16, -0.111128506, -0.169470553, 0.114710683, 0.0635468419, 0.353533144, -0.0367877935, -0.0200417675, -0.194362049])
    assert_allclose(res, res_scipy_191, atol=1e-14)
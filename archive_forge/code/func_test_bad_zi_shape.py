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
def test_bad_zi_shape(self, dt):
    x = np.empty((3, 15, 3), dt)
    sos = np.zeros((4, 6))
    zi = np.empty((4, 3, 3, 2))
    with pytest.raises(ValueError, match='should be all ones'):
        sosfilt(sos, x, zi=zi, axis=1)
    sos[:, 3] = 1.0
    with pytest.raises(ValueError, match='Invalid zi shape'):
        sosfilt(sos, x, zi=zi, axis=1)
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
@pytest.mark.parametrize('dtype', [np.bool_, np.complex64, np.complex128, np.clongdouble, np.float16])
def test_invalid_dtypes(self, dtype):
    in_typed = np.array(self.IN, dtype=dtype)
    with pytest.raises(ValueError, match='not supported'):
        signal.medfilt(in_typed)
    with pytest.raises(ValueError, match='not supported'):
        signal.medfilt2d(in_typed)
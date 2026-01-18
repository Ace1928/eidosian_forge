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
def test_lfilter_bad_object():
    if hasattr(sys, 'abiflags') and 'd' in sys.abiflags:
        pytest.skip('test is flaky when run with python3-dbg')
    assert_raises(TypeError, lfilter, [1.0], [1.0], [1.0, None, 2.0])
    assert_raises(TypeError, lfilter, [1.0], [None], [1.0, 2.0, 3.0])
    assert_raises(TypeError, lfilter, [None], [1.0], [1.0, 2.0, 3.0])
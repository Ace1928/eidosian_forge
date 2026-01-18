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
@pytest.mark.parametrize('shape_a_0, shape_b_0', gen_oa_shapes([50, 47, 6, 4, 1]))
@pytest.mark.parametrize('is_complex', [True, False])
@pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
def test_1d_noaxes(self, shape_a_0, shape_b_0, is_complex, mode, monkeypatch):
    a = np.random.rand(shape_a_0)
    b = np.random.rand(shape_b_0)
    if is_complex:
        a = a + 1j * np.random.rand(shape_a_0)
        b = b + 1j * np.random.rand(shape_b_0)
    expected = fftconvolve(a, b, mode=mode)
    monkeypatch.setattr(signal._signaltools, 'fftconvolve', fftconvolve_err)
    out = oaconvolve(a, b, mode=mode)
    assert_array_almost_equal(out, expected)
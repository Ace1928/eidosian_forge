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
def test_initial_conditions_3d_axis1(self, dt):
    x = np.random.RandomState(159).randint(0, 5, size=(2, 15, 3))
    x = x.astype(dt)
    zpk = signal.butter(6, 0.35, output='zpk')
    sos = zpk2sos(*zpk)
    nsections = sos.shape[0]
    axis = 1
    shp = list(x.shape)
    shp[axis] = 2
    shp = [nsections] + shp
    z0 = np.zeros(shp)
    yf, zf = sosfilt(sos, x, axis=axis, zi=z0)
    y1, z1 = sosfilt(sos, x[:, :5, :], axis=axis, zi=z0)
    y2, z2 = sosfilt(sos, x[:, 5:, :], axis=axis, zi=z1)
    y = np.concatenate((y1, y2), axis=axis)
    assert_allclose_cast(y, yf, rtol=1e-10, atol=1e-13)
    assert_allclose_cast(z2, zf, rtol=1e-10, atol=1e-13)
    zi = sosfilt_zi(sos)
    zi.shape = [nsections, 1, 2, 1]
    zi = zi * x[:, 0:1, :]
    y = sosfilt(sos, x, axis=axis, zi=zi)[0]
    b, a = zpk2tf(*zpk)
    zi = lfilter_zi(b, a)
    zi.shape = [1, zi.size, 1]
    zi = zi * x[:, 0:1, :]
    y_tf = lfilter(b, a, x, axis=axis, zi=zi)[0]
    assert_allclose_cast(y, y_tf, rtol=1e-10, atol=1e-13)
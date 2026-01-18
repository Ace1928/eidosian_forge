import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def test_extrap(self):
    b = _make_random_spline()
    t, c, k = b.tck
    dt = t[-1] - t[0]
    xx = np.linspace(t[k] - dt, t[-k - 1] + dt, 50)
    mask = (t[k] < xx) & (xx < t[-k - 1])
    assert_allclose(b(xx[mask], extrapolate=True), b(xx[mask], extrapolate=False))
    assert_allclose(b(xx, extrapolate=True), splev(xx, (t, c, k), ext=0))
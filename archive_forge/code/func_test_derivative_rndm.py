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
def test_derivative_rndm(self):
    b = _make_random_spline()
    t, c, k = b.tck
    xx = np.linspace(t[0], t[-1], 50)
    xx = np.r_[xx, t]
    for der in range(1, k + 1):
        yd = splev(xx, (t, c, k), der=der)
        assert_allclose(yd, b(xx, nu=der), atol=1e-14)
    assert_allclose(b(xx, nu=k + 1), 0, atol=1e-14)
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
def test_derivative_method(self):
    b = _make_random_spline(k=5)
    t, c, k = b.tck
    b0 = BSpline(t, c, k)
    xx = np.linspace(t[k], t[-k - 1], 20)
    for j in range(1, k):
        b = b.derivative()
        assert_allclose(b0(xx, j), b(xx), atol=1e-12, rtol=1e-12)
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
def test_antiderivative_method(self):
    b = _make_random_spline()
    t, c, k = b.tck
    xx = np.linspace(t[k], t[-k - 1], 20)
    assert_allclose(b.antiderivative().derivative()(xx), b(xx), atol=1e-14, rtol=1e-14)
    c = np.c_[c, c, c]
    c = np.dstack((c, c))
    b = BSpline(t, c, k)
    assert_allclose(b.antiderivative().derivative()(xx), b(xx), atol=1e-14, rtol=1e-14)
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
def test_full_matrix(self):
    np.random.seed(1234)
    k, n = (3, 7)
    x = np.sort(np.random.random(size=n))
    y = np.random.random(size=n)
    t = _not_a_knot(x, k)
    b = make_interp_spline(x, y, k, t)
    cf = make_interp_full_matr(x, y, t, k)
    assert_allclose(b.c, cf, atol=1e-14, rtol=1e-14)
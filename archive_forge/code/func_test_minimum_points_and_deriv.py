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
def test_minimum_points_and_deriv(self):
    k = 3
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    b = make_interp_spline(x, y, k, bc_type=([(1, 0.0)], [(1, 3.0)]))
    xx = np.linspace(0.0, 1.0)
    yy = xx ** 3
    assert_allclose(b(xx), yy, atol=1e-14, rtol=1e-14)
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
@pytest.mark.parametrize('bc_type', ['natural', 'clamped', 'periodic', 'not-a-knot'])
def test_from_power_basis(self, bc_type):
    np.random.seed(1234)
    x = np.sort(np.random.random(20))
    y = np.random.random(20)
    if bc_type == 'periodic':
        y[-1] = y[0]
    cb = CubicSpline(x, y, bc_type=bc_type)
    bspl = BSpline.from_power_basis(cb, bc_type=bc_type)
    xx = np.linspace(0, 1, 20)
    assert_allclose(cb(xx), bspl(xx), atol=1e-15)
    bspl_new = make_interp_spline(x, y, bc_type=bc_type)
    assert_allclose(bspl.c, bspl_new.c, atol=1e-15)
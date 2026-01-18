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
def test_from_power_basis_exmp(self):
    """
        For x = [0, 1, 2, 3, 4] and y = [1, 1, 1, 1, 1]
        the coefficients of Cubic Spline in the power basis:

        $[[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[0, 0, 0, 0, 0],\\$
        $[1, 1, 1, 1, 1]]$

        It could be shown explicitly that coefficients of the interpolating
        function in B-spline basis are c = [1, 1, 1, 1, 1, 1, 1]
        """
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 1, 1, 1, 1])
    bspl = BSpline.from_power_basis(CubicSpline(x, y, bc_type='natural'), bc_type='natural')
    assert_allclose(bspl.c, [1, 1, 1, 1, 1, 1, 1], atol=1e-15)
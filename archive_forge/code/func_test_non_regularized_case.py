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
def test_non_regularized_case(self):
    """
        In case the regularization parameter is 0, the resulting spline
        is an interpolation spline with natural boundary conditions.
        """
    np.random.seed(1234)
    n = 100
    x = np.sort(np.random.random_sample(n) * 4 - 2)
    y = x ** 2 * np.sin(4 * x) + x ** 3 + np.random.normal(0.0, 1.5, n)
    spline_GCV = make_smoothing_spline(x, y, lam=0.0)
    spline_interp = make_interp_spline(x, y, 3, bc_type='natural')
    grid = np.linspace(x[0], x[-1], 2 * n)
    assert_allclose(spline_GCV(grid), spline_interp(grid), atol=1e-15)
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
def test_2D_separable(self):
    xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
    t2, c2, k = self.make_2d_case()
    target = [x ** 3 * (y ** 3 + 2 * y) for x, y in xi]
    assert_allclose([bspline2(xy, t2, c2, k) for xy in xi], target, atol=1e-14)
    bspl2 = NdBSpline(t2, c2, k=3)
    assert bspl2(xi).shape == (len(xi),)
    assert_allclose(bspl2(xi), target, atol=1e-14)
    rng = np.random.default_rng(12345)
    xi = rng.uniform(size=(4, 3, 2)) * 5
    result = bspl2(xi)
    assert result.shape == (4, 3)
    x, y = xi.reshape((-1, 2)).T
    assert_allclose(result.ravel(), x ** 3 * (y ** 3 + 2 * y), atol=1e-14)
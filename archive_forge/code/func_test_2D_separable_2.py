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
def test_2D_separable_2(self):
    ndim = 2
    xi = [(1.5, 2.5), (2.5, 1), (0.5, 1.5)]
    target = [x ** 3 * (y ** 3 + 2 * y) for x, y in xi]
    t2, c2, k = self.make_2d_case()
    c2_4 = np.dstack((c2, c2, c2, c2))
    xy = (1.5, 2.5)
    bspl2_4 = NdBSpline(t2, c2_4, k=3)
    result = bspl2_4(xy)
    val_single = NdBSpline(t2, c2, k)(xy)
    assert result.shape == (4,)
    assert_allclose(result, [val_single] * 4, atol=1e-14)
    assert bspl2_4(xi).shape == np.shape(xi)[:-1] + bspl2_4.c.shape[ndim:]
    assert_allclose(bspl2_4(xi) - np.asarray(target)[:, None], 0, atol=5e-14)
    c2_22 = c2_4.reshape((6, 6, 2, 2))
    bspl2_22 = NdBSpline(t2, c2_22, k=3)
    result = bspl2_22(xy)
    assert result.shape == (2, 2)
    assert_allclose(result, [[val_single, val_single], [val_single, val_single]], atol=1e-14)
    assert bspl2_22(xi).shape == np.shape(xi)[:-1] + bspl2_22.c.shape[ndim:]
    assert_allclose(bspl2_22(xi) - np.asarray(target)[:, None, None], 0, atol=5e-14)
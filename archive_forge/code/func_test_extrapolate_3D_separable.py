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
@pytest.mark.parametrize('cls_extrap', [None, True])
@pytest.mark.parametrize('call_extrap', [None, True])
def test_extrapolate_3D_separable(self, cls_extrap, call_extrap):
    t3, c3, k = self.make_3d_case()
    bspl3 = NdBSpline(t3, c3, k=3, extrapolate=cls_extrap)
    x, y, z = ([-2, -1, 7], [-3, -0.5, 6.5], [-1, -1.5, 7.5])
    x, y, z = map(np.asarray, (x, y, z))
    xi = [_ for _ in zip(x, y, z)]
    target = x ** 3 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1)
    result = bspl3(xi, extrapolate=call_extrap)
    assert_allclose(result, target, atol=1e-14)
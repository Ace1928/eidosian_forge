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
def test_3D_separable(self):
    rng = np.random.default_rng(12345)
    x, y, z = rng.uniform(size=(3, 11)) * 5
    target = x ** 3 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1)
    t3, c3, k = self.make_3d_case()
    bspl3 = NdBSpline(t3, c3, k=3)
    xi = [_ for _ in zip(x, y, z)]
    result = bspl3(xi)
    assert result.shape == (11,)
    assert_allclose(result, target, atol=1e-14)
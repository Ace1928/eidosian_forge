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
def test_3D_derivative(self):
    t3, c3, k = self.make_3d_case()
    bspl3 = NdBSpline(t3, c3, k=3)
    rng = np.random.default_rng(12345)
    x, y, z = rng.uniform(size=(3, 11)) * 5
    xi = [_ for _ in zip(x, y, z)]
    assert_allclose(bspl3(xi, nu=(1, 0, 0)), 3 * x ** 2 * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1), atol=1e-14)
    assert_allclose(bspl3(xi, nu=(2, 0, 0)), 6 * x * (y ** 3 + 2 * y) * (z ** 3 + 3 * z + 1), atol=1e-14)
    assert_allclose(bspl3(xi, nu=(2, 1, 0)), 6 * x * (3 * y ** 2 + 2) * (z ** 3 + 3 * z + 1), atol=1e-14)
    assert_allclose(bspl3(xi, nu=(2, 1, 3)), 6 * x * (3 * y ** 2 + 2) * 6, atol=1e-14)
    assert_allclose(bspl3(xi, nu=(2, 1, 4)), np.zeros(len(xi)), atol=1e-14)
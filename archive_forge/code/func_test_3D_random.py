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
def test_3D_random(self):
    rng = np.random.default_rng(12345)
    k = 3
    tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
    ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
    tz = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
    c = rng.uniform(size=(tx.size - k - 1, ty.size - k - 1, tz.size - k - 1))
    spl = NdBSpline((tx, ty, tz), c, k=k)
    spl_0 = NdBSpline0((tx, ty, tz), c, k=k)
    xi = (1.0, 1.0, 1)
    assert_allclose(spl(xi), spl_0(xi), atol=1e-14)
    xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1], [0.9, 1.4, 1.9]]
    assert_allclose(spl(xi), [spl_0(xp) for xp in xi], atol=1e-14)
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
def test_2D_mixed_random(self):
    rng = np.random.default_rng(12345)
    kx, ky = (2, 3)
    tx = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=7)) * 3, 3, 3, 3, 3]
    ty = np.r_[0, 0, 0, 0, np.sort(rng.uniform(size=8)) * 4, 4, 4, 4, 4]
    c = rng.uniform(size=(tx.size - kx - 1, ty.size - ky - 1))
    xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1]]
    bspl2 = NdBSpline((tx, ty), c, k=(kx, ky))
    bspl2_0 = NdBSpline0((tx, ty), c, k=(kx, ky))
    assert_allclose(bspl2(xi), [bspl2_0(xp) for xp in xi], atol=1e-14)
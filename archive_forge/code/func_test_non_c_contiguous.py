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
def test_non_c_contiguous(self):
    rng = np.random.default_rng(12345)
    kx, ky = (3, 3)
    tx = np.sort(rng.uniform(low=0, high=4, size=16))
    tx = np.r_[(tx[0],) * kx, tx, (tx[-1],) * kx]
    ty = np.sort(rng.uniform(low=0, high=4, size=16))
    ty = np.r_[(ty[0],) * ky, ty, (ty[-1],) * ky]
    assert not tx[::2].flags.c_contiguous
    assert not ty[::2].flags.c_contiguous
    c = rng.uniform(size=(tx.size // 2 - kx - 1, ty.size // 2 - ky - 1))
    c = c.T
    assert not c.flags.c_contiguous
    xi = np.c_[[1, 1.5, 2], [1.1, 1.6, 2.1]]
    bspl2 = NdBSpline((tx[::2], ty[::2]), c, k=(kx, ky))
    bspl2_0 = NdBSpline0((tx[::2], ty[::2]), c, k=(kx, ky))
    assert_allclose(bspl2(xi), [bspl2_0(xp) for xp in xi], atol=1e-14)
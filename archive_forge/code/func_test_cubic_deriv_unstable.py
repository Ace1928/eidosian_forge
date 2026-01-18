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
@pytest.mark.xfail(reason='unstable')
def test_cubic_deriv_unstable(self):
    k = 3
    t = _augknt(self.xx, k)
    der_l = [(1, 3.0), (2, 4.0)]
    b = make_interp_spline(self.xx, self.yy, k, t, bc_type=(der_l, None))
    assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
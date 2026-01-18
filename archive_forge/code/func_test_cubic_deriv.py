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
def test_cubic_deriv(self):
    k = 3
    der_l, der_r = ([(1, 3.0)], [(1, 4.0)])
    b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
    assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
    assert_allclose([b(self.xx[0], 1), b(self.xx[-1], 1)], [der_l[0][1], der_r[0][1]], atol=1e-14, rtol=1e-14)
    der_l, der_r = ([(2, 0)], [(2, 0)])
    b = make_interp_spline(self.xx, self.yy, k, bc_type=(der_l, der_r))
    assert_allclose(b(self.xx), self.yy, atol=1e-14, rtol=1e-14)
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
def test_2D_mixed(self):
    t2, c2, kx, ky = self.make_2d_mixed()
    xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
    target = [x ** 3 * (y ** 2 + 2 * y) for x, y in xi]
    bspl2 = NdBSpline(t2, c2, k=(kx, ky))
    assert bspl2(xi).shape == (len(xi),)
    assert_allclose(bspl2(xi), target, atol=1e-14)
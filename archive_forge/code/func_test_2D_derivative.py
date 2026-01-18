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
def test_2D_derivative(self):
    t2, c2, kx, ky = self.make_2d_mixed()
    xi = [(1.4, 4.5), (2.5, 2.4), (4.5, 3.5)]
    bspl2 = NdBSpline(t2, c2, k=(kx, ky))
    der = bspl2(xi, nu=(1, 0))
    assert_allclose(der, [3 * x ** 2 * (y ** 2 + 2 * y) for x, y in xi], atol=1e-14)
    der = bspl2(xi, nu=(1, 1))
    assert_allclose(der, [3 * x ** 2 * (2 * y + 2) for x, y in xi], atol=1e-14)
    der = bspl2(xi, nu=(0, 0))
    assert_allclose(der, [x ** 3 * (y ** 2 + 2 * y) for x, y in xi], atol=1e-14)
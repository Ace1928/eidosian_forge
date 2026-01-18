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
def test_knots_multiplicity():

    def check_splev(b, j, der=0, atol=1e-14, rtol=1e-14):
        t, c, k = b.tck
        x = np.unique(t)
        x = np.r_[t[0] - 0.1, 0.5 * (x[1:] + x[:1]), t[-1] + 0.1]
        assert_allclose(splev(x, (t, c, k), der), b(x, der), atol=atol, rtol=rtol, err_msg=f'der = {der}  k = {b.k}')
    for k in [1, 2, 3, 4, 5]:
        b = _make_random_spline(k=k)
        for j, b1 in enumerate(_make_multiples(b)):
            check_splev(b1, j)
            for der in range(1, k + 1):
                check_splev(b1, j, der, 1e-12, 1e-12)
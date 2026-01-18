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
def make_lsq_full_matrix(x, y, t, k=3):
    """Make the least-square spline, full matrices."""
    x, y, t = map(np.asarray, (x, y, t))
    m = x.size
    n = t.size - k - 1
    A = np.zeros((m, n), dtype=np.float64)
    for j in range(m):
        xval = x[j]
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        A[j, left - k:left + 1] = bb
    B = np.dot(A.T, A)
    Y = np.dot(A.T, y)
    c = sl.solve(B, Y)
    return (c, (A, Y))
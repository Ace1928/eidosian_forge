import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
def test_gaussian_truncate():
    arr = numpy.zeros((100, 100), float)
    arr[50, 50] = 1
    num_nonzeros_2 = (ndimage.gaussian_filter(arr, 5, truncate=2) > 0).sum()
    assert_equal(num_nonzeros_2, 21 ** 2)
    num_nonzeros_5 = (ndimage.gaussian_filter(arr, 5, truncate=5) > 0).sum()
    assert_equal(num_nonzeros_5, 51 ** 2)
    f = ndimage.gaussian_filter(arr, [0.5, 2.5], truncate=3.5)
    fpos = f > 0
    n0 = fpos.any(axis=0).sum()
    assert_equal(n0, 19)
    n1 = fpos.any(axis=1).sum()
    assert_equal(n1, 5)
    x = numpy.zeros(51)
    x[25] = 1
    f = ndimage.gaussian_filter1d(x, sigma=2, truncate=3.5)
    n = (f > 0).sum()
    assert_equal(n, 15)
    y = ndimage.gaussian_laplace(x, sigma=2, truncate=3.5)
    nonzero_indices = numpy.nonzero(y != 0)[0]
    n = numpy.ptp(nonzero_indices) + 1
    assert_equal(n, 15)
    y = ndimage.gaussian_gradient_magnitude(x, sigma=2, truncate=3.5)
    nonzero_indices = numpy.nonzero(y != 0)[0]
    n = numpy.ptp(nonzero_indices) + 1
    assert_equal(n, 15)
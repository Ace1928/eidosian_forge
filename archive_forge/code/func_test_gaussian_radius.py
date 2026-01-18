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
def test_gaussian_radius():
    x = numpy.zeros(7)
    x[3] = 1
    f1 = ndimage.gaussian_filter1d(x, sigma=2, truncate=1.5)
    f2 = ndimage.gaussian_filter1d(x, sigma=2, radius=3)
    assert_equal(f1, f2)
    a = numpy.zeros((9, 9))
    a[4, 4] = 1
    f1 = ndimage.gaussian_filter(a, sigma=0.5, truncate=3.5)
    f2 = ndimage.gaussian_filter(a, sigma=0.5, radius=2)
    assert_equal(f1, f2)
    a = numpy.zeros((50, 50))
    a[25, 25] = 1
    f1 = ndimage.gaussian_filter(a, sigma=[0.5, 2.5], truncate=3.5)
    f2 = ndimage.gaussian_filter(a, sigma=[0.5, 2.5], radius=[2, 9])
    assert_equal(f1, f2)
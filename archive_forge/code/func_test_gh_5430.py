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
def test_gh_5430():
    sigma = numpy.int32(1)
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert_equal(out, [sigma])
    sigma = numpy.int64(1)
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert_equal(out, [sigma])
    sigma = 1
    out = ndimage._ni_support._normalize_sequence(sigma, 1)
    assert_equal(out, [sigma])
    sigma = [1, 1]
    out = ndimage._ni_support._normalize_sequence(sigma, 2)
    assert_equal(out, sigma)
    x = numpy.random.normal(size=(256, 256))
    perlin = numpy.zeros_like(x)
    for i in 2 ** numpy.arange(6):
        perlin += ndimage.gaussian_filter(x, i, mode='wrap') * i ** 2
    x = numpy.int64(21)
    ndimage._ni_support._normalize_sequence(x, 0)
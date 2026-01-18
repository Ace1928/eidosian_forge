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
def test_correlate07(self):
    array = numpy.array([1, 2, 3])
    expected = [5, 8, 11]
    weights = numpy.array([1, 2, 1])
    output = ndimage.correlate(array, weights)
    assert_array_almost_equal(output, expected)
    output = ndimage.convolve(array, weights)
    assert_array_almost_equal(output, expected)
    output = ndimage.correlate1d(array, weights)
    assert_array_almost_equal(output, expected)
    output = ndimage.convolve1d(array, weights)
    assert_array_almost_equal(output, expected)
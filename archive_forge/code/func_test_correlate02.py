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
def test_correlate02(self):
    array = numpy.array([1, 2, 3])
    kernel = numpy.array([1])
    output = ndimage.correlate(array, kernel)
    assert_array_almost_equal(array, output)
    output = ndimage.convolve(array, kernel)
    assert_array_almost_equal(array, output)
    output = ndimage.correlate1d(array, kernel)
    assert_array_almost_equal(array, output)
    output = ndimage.convolve1d(array, kernel)
    assert_array_almost_equal(array, output)
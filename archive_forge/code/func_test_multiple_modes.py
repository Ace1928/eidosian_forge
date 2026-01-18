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
def test_multiple_modes():
    arr = numpy.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mode1 = 'reflect'
    mode2 = ['reflect', 'reflect']
    assert_equal(ndimage.gaussian_filter(arr, 1, mode=mode1), ndimage.gaussian_filter(arr, 1, mode=mode2))
    assert_equal(ndimage.prewitt(arr, mode=mode1), ndimage.prewitt(arr, mode=mode2))
    assert_equal(ndimage.sobel(arr, mode=mode1), ndimage.sobel(arr, mode=mode2))
    assert_equal(ndimage.laplace(arr, mode=mode1), ndimage.laplace(arr, mode=mode2))
    assert_equal(ndimage.gaussian_laplace(arr, 1, mode=mode1), ndimage.gaussian_laplace(arr, 1, mode=mode2))
    assert_equal(ndimage.maximum_filter(arr, size=5, mode=mode1), ndimage.maximum_filter(arr, size=5, mode=mode2))
    assert_equal(ndimage.minimum_filter(arr, size=5, mode=mode1), ndimage.minimum_filter(arr, size=5, mode=mode2))
    assert_equal(ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode1), ndimage.gaussian_gradient_magnitude(arr, 1, mode=mode2))
    assert_equal(ndimage.uniform_filter(arr, 5, mode=mode1), ndimage.uniform_filter(arr, 5, mode=mode2))
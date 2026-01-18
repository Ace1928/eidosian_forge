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
def test_correlate11(self):
    array = numpy.array([[1, 2, 3], [4, 5, 6]])
    kernel = numpy.array([[1, 1], [1, 1]])
    output = ndimage.correlate(array, kernel)
    assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
    output = ndimage.convolve(array, kernel)
    assert_array_almost_equal([[12, 16, 18], [18, 22, 24]], output)
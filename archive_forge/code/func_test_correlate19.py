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
@pytest.mark.parametrize('dtype_array', types)
def test_correlate19(self, dtype_array):
    kernel = numpy.array([[1, 0], [0, 1]])
    array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype_array)
    output = ndimage.correlate(array, kernel, output=numpy.float32, mode='nearest', origin=[-1, 0])
    assert_array_almost_equal([[5, 6, 8], [8, 9, 11]], output)
    assert_equal(output.dtype.type, numpy.float32)
    output = ndimage.convolve(array, kernel, output=numpy.float32, mode='nearest', origin=[-1, 0])
    assert_array_almost_equal([[3, 5, 6], [6, 8, 9]], output)
    assert_equal(output.dtype.type, numpy.float32)
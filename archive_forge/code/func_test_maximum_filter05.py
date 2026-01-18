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
def test_maximum_filter05(self):
    array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
    filter_shape = numpy.array([2, 3])
    output = ndimage.maximum_filter(array, filter_shape)
    assert_array_almost_equal([[3, 5, 5, 5, 4], [7, 9, 9, 9, 5], [8, 9, 9, 9, 7]], output)
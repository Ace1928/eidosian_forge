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
@pytest.mark.parametrize('mode, expected_value', [('nearest', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]), ('wrap', [[5, 6, 4], [8, 9, 7], [2, 3, 1]]), ('reflect', [[5, 6, 6], [8, 9, 9], [8, 9, 9]]), ('mirror', [[5, 6, 5], [8, 9, 8], [5, 6, 5]]), ('constant', [[5, 6, 0], [8, 9, 0], [0, 0, 0]])])
def test_extend06(self, mode, expected_value):
    array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    weights = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    output = ndimage.correlate(array, weights, mode=mode, cval=0)
    assert_array_equal(output, expected_value)
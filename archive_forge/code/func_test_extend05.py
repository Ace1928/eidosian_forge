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
@pytest.mark.parametrize('mode, expected_value', [('nearest', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]), ('wrap', [[9, 7, 8], [3, 1, 2], [6, 4, 5]]), ('reflect', [[1, 1, 2], [1, 1, 2], [4, 4, 5]]), ('mirror', [[5, 4, 5], [2, 1, 2], [5, 4, 5]]), ('constant', [[0, 0, 0], [0, 1, 2], [0, 4, 5]])])
def test_extend05(self, mode, expected_value):
    array = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    weights = numpy.array([[1, 0], [0, 0]])
    output = ndimage.correlate(array, weights, mode=mode, cval=0)
    assert_array_equal(output, expected_value)
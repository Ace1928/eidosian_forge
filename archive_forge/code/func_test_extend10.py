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
@pytest.mark.parametrize('mode, expected_value', [('nearest', [[3], [3], [3]]), ('wrap', [[2], [3], [1]]), ('reflect', [[2], [1], [1]]), ('mirror', [[1], [2], [3]]), ('constant', [[0], [0], [0]])])
def test_extend10(self, mode, expected_value):
    array = numpy.array([[1], [2], [3]])
    weights = numpy.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
    output = ndimage.correlate(array, weights, mode=mode, cval=0)
    assert_array_equal(output, expected_value)
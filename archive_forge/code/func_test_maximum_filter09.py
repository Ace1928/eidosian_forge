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
def test_maximum_filter09(self):
    array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
    footprint = [[1, 0, 1], [1, 1, 0]]
    output = ndimage.maximum_filter(array, footprint=footprint, origin=[-1, 0])
    assert_array_almost_equal([[7, 7, 9, 9, 5], [7, 9, 8, 9, 7], [8, 8, 8, 7, 7]], output)
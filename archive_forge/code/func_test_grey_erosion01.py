import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_grey_erosion01(self):
    array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
    footprint = [[1, 0, 1], [1, 1, 0]]
    output = ndimage.grey_erosion(array, footprint=footprint)
    assert_array_almost_equal([[2, 2, 1, 1, 1], [2, 3, 1, 3, 1], [5, 5, 3, 3, 1]], output)
import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_grey_closing02(self):
    array = numpy.array([[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]])
    footprint = [[1, 0, 1], [1, 1, 0]]
    structure = [[0, 0, 0], [0, 0, 0]]
    tmp = ndimage.grey_dilation(array, footprint=footprint, structure=structure)
    expected = ndimage.grey_erosion(tmp, footprint=footprint, structure=structure)
    output = ndimage.grey_closing(array, footprint=footprint, structure=structure)
    assert_array_almost_equal(expected, output)
import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_binary_erosion38(self):
    data = numpy.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    iterations = 2.0
    with assert_raises(TypeError):
        _ = ndimage.binary_erosion(data, iterations=iterations)
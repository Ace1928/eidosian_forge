import numpy
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from . import types
def test_binary_dilation30(self):
    struct = [[0, 1], [1, 1]]
    expected = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
    data = numpy.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], bool)
    out = numpy.zeros(data.shape, bool)
    ndimage.binary_dilation(data, struct, iterations=2, output=out)
    assert_array_almost_equal(out, expected)
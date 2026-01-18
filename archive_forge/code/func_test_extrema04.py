import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_extrema04():
    labels = [1, 2, 0, 4]
    for type in types:
        input = np.array([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output1 = ndimage.extrema(input, labels, [1, 2])
        output2 = ndimage.minimum(input, labels, [1, 2])
        output3 = ndimage.maximum(input, labels, [1, 2])
        output4 = ndimage.minimum_position(input, labels, [1, 2])
        output5 = ndimage.maximum_position(input, labels, [1, 2])
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)
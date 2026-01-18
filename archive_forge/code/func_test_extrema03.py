import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_extrema03():
    labels = np.array([[1, 2], [2, 3]])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output1 = ndimage.extrema(input, labels=labels, index=[2, 3, 8])
        output2 = ndimage.minimum(input, labels=labels, index=[2, 3, 8])
        output3 = ndimage.maximum(input, labels=labels, index=[2, 3, 8])
        output4 = ndimage.minimum_position(input, labels=labels, index=[2, 3, 8])
        output5 = ndimage.maximum_position(input, labels=labels, index=[2, 3, 8])
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)
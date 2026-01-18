import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_mean04():
    labels = np.array([[1, 2], [2, 4]], np.int8)
    with np.errstate(all='ignore'):
        for type in types:
            input = np.array([[1, 2], [3, 4]], type)
            output = ndimage.mean(input, labels=labels, index=[4, 8, 2])
            assert_array_almost_equal(output[[0, 2]], [4.0, 2.5])
            assert_(np.isnan(output[1]))
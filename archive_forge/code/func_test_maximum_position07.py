import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_maximum_position07():
    labels = np.array([1.0, 2.5, 0.0, 4.5])
    for type in types:
        input = np.array([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels, [1.0, 4.5])
        assert_equal(output[0], (0, 0))
        assert_equal(output[1], (0, 3))
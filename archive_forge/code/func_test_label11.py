import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label11():
    for type in types:
        data = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0]], type)
        out, n = ndimage.label(data)
        expected = [[1, 0, 0, 0, 0, 0], [0, 0, 2, 2, 0, 0], [0, 0, 2, 2, 2, 0], [3, 3, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0], [0, 0, 0, 4, 4, 0]]
        assert_array_almost_equal(out, expected)
        assert_equal(n, 4)
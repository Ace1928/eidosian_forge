import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label09():
    data = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0]])
    struct = ndimage.generate_binary_structure(2, 2)
    out, n = ndimage.label(data, struct)
    assert_array_almost_equal(out, [[1, 0, 0, 0, 0, 0], [0, 0, 2, 2, 0, 0], [0, 0, 2, 2, 2, 0], [2, 2, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0], [0, 0, 0, 3, 3, 0]])
    assert_equal(n, 3)
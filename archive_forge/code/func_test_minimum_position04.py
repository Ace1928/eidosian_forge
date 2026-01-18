import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_minimum_position04():
    input = np.array([[5, 4, 2, 5], [3, 7, 1, 2], [1, 5, 1, 1]], bool)
    output = ndimage.minimum_position(input)
    assert_equal(output, (0, 0))
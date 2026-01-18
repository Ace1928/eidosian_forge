import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_minimum_position02():
    for type in types:
        input = np.array([[5, 4, 2, 5], [3, 7, 0, 2], [1, 5, 1, 1]], type)
        output = ndimage.minimum_position(input)
        assert_equal(output, (1, 2))
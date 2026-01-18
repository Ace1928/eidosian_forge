import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_histogram02():
    labels = [1, 1, 1, 1, 2, 2, 2, 2]
    expected = [0, 2, 0, 1, 1]
    input = np.array([1, 1, 3, 4, 3, 3, 3, 3])
    output = ndimage.histogram(input, 0, 4, 5, labels, 1)
    assert_array_almost_equal(output, expected)
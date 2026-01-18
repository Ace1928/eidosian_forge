import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_mean02():
    labels = np.array([1, 0], bool)
    input = np.array([[1, 2], [3, 4]], bool)
    output = ndimage.mean(input, labels=labels)
    assert_almost_equal(output, 1.0)
import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_maximum_position01():
    labels = np.array([1, 0], bool)
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.maximum_position(input, labels=labels)
        assert_equal(output, (1, 0))
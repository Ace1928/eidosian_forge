import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_maximum04():
    labels = np.array([[1, 2], [2, 3]])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.maximum(input, labels=labels, index=[2, 3, 8])
        assert_array_almost_equal(output, [3.0, 4.0, 0.0])
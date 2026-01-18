import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_minimum03():
    labels = np.array([1, 2])
    for type in types:
        input = np.array([[1, 2], [3, 4]], type)
        output = ndimage.minimum(input, labels=labels, index=2)
        assert_almost_equal(output, 2.0)
import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_histogram01():
    expected = np.ones(10)
    input = np.arange(10)
    output = ndimage.histogram(input, 0, 10, 10)
    assert_array_almost_equal(output, expected)
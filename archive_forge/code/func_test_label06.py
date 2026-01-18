import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label06():
    data = np.array([1, 0, 1, 1, 0, 1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1, 0, 2, 2, 0, 3])
    assert_equal(n, 3)
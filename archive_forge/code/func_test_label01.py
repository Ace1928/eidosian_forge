import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label01():
    data = np.ones([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 1)
    assert_equal(n, 1)
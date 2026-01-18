import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_label02():
    data = np.zeros([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 0)
    assert_equal(n, 0)
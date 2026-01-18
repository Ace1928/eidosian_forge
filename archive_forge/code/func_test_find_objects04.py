import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_find_objects04():
    data = np.zeros([1], dtype=int)
    out = ndimage.find_objects(data)
    assert_equal(out, [])
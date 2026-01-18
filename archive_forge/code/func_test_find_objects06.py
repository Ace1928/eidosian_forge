import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_find_objects06():
    data = np.array([1, 0, 2, 2, 0, 3])
    out = ndimage.find_objects(data)
    assert_equal(out, [(slice(0, 1, None),), (slice(2, 4, None),), (slice(5, 6, None),)])
import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_find_objects09():
    data = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 2, 2, 0, 0], [0, 0, 2, 2, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 0]])
    out = ndimage.find_objects(data)
    assert_equal(out, [(slice(0, 1, None), slice(0, 1, None)), (slice(1, 3, None), slice(2, 5, None)), None, (slice(5, 6, None), slice(3, 5, None))])
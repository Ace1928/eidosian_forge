import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_maximum05():
    x = np.array([-3, -2, -1])
    assert_equal(ndimage.maximum(x), -1)
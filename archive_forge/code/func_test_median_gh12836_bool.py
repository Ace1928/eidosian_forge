import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_median_gh12836_bool():
    a = np.asarray([1, 1], dtype=bool)
    output = ndimage.median(a, labels=np.ones((2,)), index=[1])
    assert_array_almost_equal(output, [1.0])
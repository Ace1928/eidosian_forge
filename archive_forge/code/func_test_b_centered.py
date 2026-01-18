import os.path
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
def test_b_centered(self):
    x = [0, 1, 2, 6]
    labels = [0, 0, 9, 9]
    index = [0, 9]
    for shp in [(4,), (2, 2)]:
        x = np.array(x).reshape(shp)
        labels = np.array(labels).reshape(shp)
        counts, sums, centers = ndimage._measurements._stats(x, labels=labels, index=index, centered=True)
        assert_array_equal(counts, [2, 2])
        assert_array_equal(sums, [1.0, 8.0])
        assert_array_equal(centers, [0.5, 8.0])
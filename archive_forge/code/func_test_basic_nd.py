import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_basic_nd():
    for dimension in (3, 4, 5):
        shape = (5,) * dimension
        hypercube = np.zeros(shape)
        slice_mid = tuple((slice(1, -1, None) for dim in range(dimension)))
        hypercube[slice_mid] = 1
        filled = flood_fill(hypercube, (2,) * dimension, 2)
        assert filled.sum() == 3 ** dimension * 2
        np.testing.assert_equal(filled, np.pad(np.ones((3,) * dimension) * 2, 1, 'constant'))
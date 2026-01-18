import numpy as np
import pytest
from skimage.util import slice_along_axes
def test_nd_crop():
    data = rng.random((50, 50, 50))
    out = slice_along_axes(data, [(0, 25)], axes=[2])
    np.testing.assert_array_equal(out, data[:, :, :25])
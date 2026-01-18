import numpy as np
import pytest
from skimage.util import slice_along_axes
def test_2d_crop_1():
    data = rng.random((50, 50))
    out = slice_along_axes(data, [(0, 25), (0, 10)])
    np.testing.assert_array_equal(out, data[:25, :10])
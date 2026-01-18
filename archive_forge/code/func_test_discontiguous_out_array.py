import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
def test_discontiguous_out_array():
    image = np.array([[5, 6, 2], [7, 2, 2], [3, 5, 1]], np.uint8)
    out_array_big = np.zeros((5, 5), np.uint8)
    out_array = out_array_big[::2, ::2]
    expected_dilation = np.array([[7, 0, 6, 0, 6], [0, 0, 0, 0, 0], [7, 0, 7, 0, 2], [0, 0, 0, 0, 0], [7, 0, 5, 0, 5]], np.uint8)
    expected_erosion = np.array([[5, 0, 2, 0, 2], [0, 0, 0, 0, 0], [2, 0, 2, 0, 1], [0, 0, 0, 0, 0], [3, 0, 1, 0, 1]], np.uint8)
    gray.dilation(image, out=out_array)
    assert_array_equal(out_array_big, expected_dilation)
    gray.erosion(image, out=out_array)
    assert_array_equal(out_array_big, expected_erosion)
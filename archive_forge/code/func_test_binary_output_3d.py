import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal
from scipy import ndimage as ndi
from skimage import data, color, morphology
from skimage.util import img_as_bool
from skimage.morphology import binary, footprints, gray
def test_binary_output_3d():
    image = np.zeros((9, 9, 9), np.uint16)
    image[2:-2, 2:-2, 2:-2] = 2 ** 14
    image[3:-3, 3:-3, 3:-3] = 2 ** 15
    image[4, 4, 4] = 2 ** 16 - 1
    bin_opened = binary.binary_opening(image)
    bin_closed = binary.binary_closing(image)
    int_opened = np.empty_like(image, dtype=np.uint8)
    int_closed = np.empty_like(image, dtype=np.uint8)
    binary.binary_opening(image, out=int_opened)
    binary.binary_closing(image, out=int_closed)
    assert_equal(bin_opened.dtype, bool)
    assert_equal(bin_closed.dtype, bool)
    assert_equal(int_opened.dtype, np.uint8)
    assert_equal(int_closed.dtype, np.uint8)
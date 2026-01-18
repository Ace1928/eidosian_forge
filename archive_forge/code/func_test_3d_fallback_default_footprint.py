import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
def test_3d_fallback_default_footprint():
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1
    opened = gray.opening(image)
    image_expected = np.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    assert_array_equal(opened, image_expected)
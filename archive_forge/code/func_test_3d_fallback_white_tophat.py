import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
def test_3d_fallback_white_tophat():
    image = np.zeros((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 1
    image[3, 2:5, 2:5] = 1
    image[4, 3:5, 3:5] = 1
    with expected_warnings(['operator.*deprecated|\\A\\Z']):
        new_image = gray.white_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    with expected_warnings(['operator.*deprecated|\\A\\Z']):
        image_expected = ndi.white_tophat(image.view(dtype=np.uint8), footprint=footprint)
    assert_array_equal(new_image, image_expected)
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
def test_hog_image_size_cell_size_mismatch():
    image = data.camera()[:150, :200]
    fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L1')
    assert len(fd) == 9 * (150 // 8) * (200 // 8)
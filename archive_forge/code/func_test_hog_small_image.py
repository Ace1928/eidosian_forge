import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
def test_hog_small_image():
    """Test that an exception is thrown whenever the input image is
    too small for the given parameters.
    """
    img = np.zeros((24, 24))
    feature.hog(img, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
    img = np.zeros((23, 23))
    with pytest.raises(ValueError, match='.*image is too small given'):
        feature.hog(img, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
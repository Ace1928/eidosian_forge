import numpy as np
import pytest
from numpy import sqrt, ceil
from numpy.testing import assert_almost_equal
from skimage import data
from skimage import img_as_float
from skimage.feature import daisy
def test_daisy_color_image_unsupported_error():
    img = np.zeros((20, 20, 3))
    with pytest.raises(ValueError):
        daisy(img)
import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
def test_image_less_than_mask():
    """Test reconstruction where the image is uniform and less than mask"""
    image = np.ones((5, 5))
    mask = np.ones((5, 5)) * 2
    assert_array_almost_equal(reconstruction(image, mask), 1)
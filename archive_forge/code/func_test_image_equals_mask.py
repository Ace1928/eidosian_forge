import math
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from skimage._shared.utils import _supported_float_type
from skimage.morphology.grayreconstruct import reconstruction
def test_image_equals_mask():
    """Test reconstruction where the image and mask are the same"""
    assert_array_almost_equal(reconstruction(np.ones((7, 5)), np.ones((7, 5))), 1)
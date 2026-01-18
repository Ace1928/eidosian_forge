import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_00_00_zeros(self):
    """Test that the Canny filter finds no points for a blank field"""
    result = feature.canny(np.zeros((20, 20)), 4, 0, 0, np.ones((20, 20), bool))
    self.assertFalse(np.any(result))
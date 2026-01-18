import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_00_01_zeros_mask(self):
    """Test that the Canny filter finds no points in a masked image"""
    result = feature.canny(np.random.uniform(size=(20, 20)), 4, 0, 0, np.zeros((20, 20), bool))
    self.assertFalse(np.any(result))
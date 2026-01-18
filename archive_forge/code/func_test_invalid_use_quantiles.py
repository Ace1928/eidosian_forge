import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_invalid_use_quantiles(self):
    image = img_as_float(data.camera()[::50, ::50])
    self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=0.5, high_threshold=3.6)
    self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=-5, high_threshold=0.5)
    self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=99, high_threshold=0.9)
    self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=0.5, high_threshold=-100)
    image = data.camera()
    self.assertRaises(ValueError, feature.canny, image, use_quantiles=True, low_threshold=50, high_threshold=150)
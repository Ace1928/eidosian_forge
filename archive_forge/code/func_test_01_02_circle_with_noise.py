import unittest
import numpy as np
import pytest
from skimage._shared.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data, feature
from skimage.util import img_as_float
def test_01_02_circle_with_noise(self):
    """Test that the Canny filter finds the circle outlines
        in a noisy image"""
    np.random.seed(0)
    i, j = np.mgrid[-200:200, -200:200].astype(float) / 200
    c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
    cf = c.astype(float) * 0.5 + np.random.uniform(size=c.shape) * 0.5
    result = feature.canny(cf, 4, 0.1, 0.2, np.ones(c.shape, bool))
    cd = binary_dilation(c, iterations=4)
    ce = binary_erosion(c, iterations=4)
    cde = np.logical_and(cd, np.logical_not(ce))
    self.assertTrue(np.all(cde[result]))
    point_count = np.sum(result)
    self.assertTrue(point_count > 1200)
    self.assertTrue(point_count < 1600)
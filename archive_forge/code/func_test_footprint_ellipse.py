import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_footprint_ellipse(self):
    """Test ellipse footprints"""
    expected_mask1 = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=np.uint8)
    actual_mask1 = footprints.ellipse(5, 3)
    expected_mask2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    actual_mask2 = footprints.ellipse(1, 1)
    assert_equal(expected_mask1, actual_mask1)
    assert_equal(expected_mask2, actual_mask2)
    assert_equal(expected_mask1, footprints.ellipse(3, 5).T)
    assert_equal(expected_mask2, footprints.ellipse(1, 1).T)
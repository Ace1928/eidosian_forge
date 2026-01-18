import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage.feature.util import (
def test_mask_border_keypoints():
    keypoints = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    np.testing.assert_equal(_mask_border_keypoints((10, 10), keypoints, 0), [1, 1, 1, 1, 1])
    np.testing.assert_equal(_mask_border_keypoints((10, 10), keypoints, 2), [0, 0, 1, 1, 1])
    np.testing.assert_equal(_mask_border_keypoints((4, 4), keypoints, 2), [0, 0, 1, 0, 0])
    np.testing.assert_equal(_mask_border_keypoints((10, 10), keypoints, 5), [0, 0, 0, 0, 0])
    np.testing.assert_equal(_mask_border_keypoints((10, 10), keypoints, 4), [0, 0, 0, 0, 1])
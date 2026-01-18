import pytest
import copy
import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage import data
from skimage.feature import BRIEF, corner_peaks, corner_harris
from skimage._shared import testing
def test_color_image_unsupported_error():
    """Brief descriptors can be evaluated on gray-scale images only."""
    img = np.zeros((20, 20, 3))
    keypoints = np.asarray([[7, 5], [11, 13]])
    with testing.raises(ValueError):
        BRIEF().extract(img, keypoints)
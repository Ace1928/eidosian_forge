import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import run_in_parallel
from skimage._shared import testing
from skimage.transform import rescale
def test_keypoints_censure_moon_image_dob():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for DoB filter."""
    detector = CENSURE()
    detector.detect(img)
    expected_keypoints = np.array([[21, 497], [36, 46], [119, 350], [185, 177], [287, 250], [357, 239], [463, 116], [464, 132], [467, 260]])
    expected_scales = np.array([3, 4, 4, 2, 2, 3, 2, 2, 2])
    assert_array_equal(expected_keypoints, detector.keypoints)
    assert_array_equal(expected_scales, detector.scales)
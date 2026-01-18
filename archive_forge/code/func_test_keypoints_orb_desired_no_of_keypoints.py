import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data
from skimage._shared.testing import run_in_parallel, xfail, arch32
from skimage.feature import ORB
from skimage.util.dtype import _convert
@run_in_parallel()
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint8', 'uint16', 'int64'])
def test_keypoints_orb_desired_no_of_keypoints(dtype):
    _img = _convert(img, dtype)
    detector_extractor = ORB(n_keypoints=10, fast_n=12, fast_threshold=0.2)
    detector_extractor.detect(_img)
    exp_rows = np.array([141.0, 108.0, 214.56, 131.0, 214.272, 67.0, 206.0, 177.0, 108.0, 141.0])
    exp_cols = np.array([323.0, 328.0, 282.24, 292.0, 281.664, 85.0, 260.0, 284.0, 328.8, 267.0])
    exp_scales = np.array([1, 1, 1.44, 1, 1.728, 1, 1, 1, 1.2, 1])
    exp_orientations = np.array([-53.97446153, 59.5055285, -96.01885186, -149.70789506, -94.70171899, -45.76429535, -51.49752849, 113.57081195, 63.30428063, -79.56091118])
    exp_response = np.array([1.01168357, 0.82934145, 0.67784179, 0.57176438, 0.56637459, 0.52248355, 0.43696175, 0.42992376, 0.37700486, 0.36126832])
    if np.dtype(dtype) == np.float32:
        assert detector_extractor.scales.dtype == np.float32
        assert detector_extractor.responses.dtype == np.float32
        assert detector_extractor.orientations.dtype == np.float32
    else:
        assert detector_extractor.scales.dtype == np.float64
        assert detector_extractor.responses.dtype == np.float64
        assert detector_extractor.orientations.dtype == np.float64
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])
    assert_almost_equal(exp_scales, detector_extractor.scales)
    assert_almost_equal(exp_response, detector_extractor.responses, 5)
    assert_almost_equal(exp_orientations, np.rad2deg(detector_extractor.orientations), 4)
    detector_extractor.detect_and_extract(img)
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])
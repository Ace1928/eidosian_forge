import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data
from skimage._shared.testing import run_in_parallel
from skimage.feature import SIFT
from skimage.util.dtype import _convert
def test_descriptor_sift():
    detector_extractor = SIFT(n_hist=2, n_ori=4)
    exp_descriptors = np.array([[173, 30, 55, 32, 173, 16, 45, 82, 173, 154, 170, 173, 173, 169, 65, 110], [173, 30, 55, 32, 173, 16, 45, 82, 173, 154, 170, 173, 173, 169, 65, 110], [189, 52, 18, 18, 189, 11, 21, 55, 189, 75, 173, 91, 189, 65, 189, 162], [172, 156, 185, 66, 92, 76, 78, 185, 185, 87, 88, 82, 98, 56, 96, 185], [216, 19, 40, 9, 196, 7, 57, 36, 216, 56, 158, 29, 216, 42, 144, 154], [169, 120, 169, 91, 129, 108, 169, 67, 169, 142, 111, 95, 169, 120, 69, 41], [199, 10, 138, 44, 178, 11, 161, 34, 199, 113, 73, 64, 199, 82, 31, 178], [154, 56, 154, 49, 144, 154, 154, 78, 154, 51, 154, 83, 154, 154, 154, 72], [230, 46, 47, 21, 230, 15, 65, 95, 230, 52, 72, 51, 230, 19, 59, 130], [155, 117, 154, 102, 155, 155, 90, 110, 145, 127, 155, 50, 57, 155, 155, 70]], dtype=np.uint8)
    detector_extractor.detect_and_extract(img)
    assert_equal(exp_descriptors, detector_extractor.descriptors[:10])
    keypoints_count = detector_extractor.keypoints.shape[0]
    assert keypoints_count == detector_extractor.descriptors.shape[0]
    assert keypoints_count == detector_extractor.orientations.shape[0]
    assert keypoints_count == detector_extractor.octaves.shape[0]
    assert keypoints_count == detector_extractor.positions.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]
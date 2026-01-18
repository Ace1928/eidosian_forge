import numpy as np
from skimage._shared.testing import assert_equal
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import BRIEF, match_descriptors, corner_peaks, corner_harris
from skimage._shared import testing
def test_binary_descriptors_rotation_crosscheck_true():
    """Verify matched keypoints and their corresponding masks results between
    image and its rotated version with the expected keypoint pairs with
    cross_check enabled."""
    img = data.astronaut()
    img = rgb2gray(img)
    tform = transform.SimilarityTransform(scale=1, rotation=0.15, translation=(0, 0))
    rotated_img = transform.warp(img, tform, clip=False)
    extractor = BRIEF(descriptor_size=512)
    keypoints1 = corner_peaks(corner_harris(img), min_distance=5, threshold_abs=0, threshold_rel=0.1)
    extractor.extract(img, keypoints1)
    descriptors1 = extractor.descriptors
    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5, threshold_abs=0, threshold_rel=0.1)
    extractor.extract(rotated_img, keypoints2)
    descriptors2 = extractor.descriptors
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
    exp_matches1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 38, 41, 42])
    exp_matches2 = np.array([0, 2, 1, 3, 4, 5, 7, 8, 14, 9, 11, 13, 23, 15, 16, 22, 17, 19, 18, 24, 27, 30, 25, 26, 28, 35, 37, 42, 29, 38, 33, 40, 36, 43, 41, 6])
    assert_equal(matches[:, 0], exp_matches1)
    assert_equal(matches[:, 1], exp_matches2)
import pytest
import copy
import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage import data
from skimage.feature import BRIEF, corner_peaks, corner_harris
from skimage._shared import testing
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'uint8', 'int'])
def test_border(dtype):
    img = np.zeros((100, 100), dtype=dtype)
    keypoints = np.array([[1, 1], [20, 20], [50, 50], [80, 80]])
    extractor = BRIEF(patch_size=41, rng=1)
    extractor.extract(img, keypoints)
    assert extractor.descriptors.shape[0] == 3
    assert_array_equal(extractor.mask, (False, True, True, True))
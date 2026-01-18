import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import run_in_parallel
from skimage._shared import testing
from skimage.transform import rescale
def test_keypoints_censure_scale_range_error():
    """Difference between the the max_scale and min_scale parameters in
    keypoints_censure should be greater than or equal to two."""
    with testing.raises(ValueError):
        CENSURE(min_scale=1, max_scale=2)
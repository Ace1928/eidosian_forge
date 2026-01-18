import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import run_in_parallel
from skimage._shared import testing
from skimage.transform import rescale
def test_keypoints_censure_mode_validity_error():
    """Mode argument in keypoints_censure can be either DoB, Octagon or
    STAR."""
    with testing.raises(ValueError):
        CENSURE(mode='dummy')
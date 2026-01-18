import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_45deg_left_upward():
    prof = profile_line(image, (8, 8), (2, 2), order=1, mode='constant')
    expected_prof = np.arange(88, 21, -22.0 / 3)
    assert_almost_equal(prof, expected_prof)
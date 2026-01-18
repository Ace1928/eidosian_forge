import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_45deg_right_upward():
    prof = profile_line(image, (8, 2), (2, 8), order=1, mode='constant')
    expected_prof = np.arange(82, 27, -6)
    assert_almost_equal(prof, expected_prof)
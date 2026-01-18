import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_45deg_left_downward():
    prof = profile_line(image, (2, 8), (8, 2), order=1, mode='constant')
    expected_prof = np.arange(28, 83, 6)
    assert_almost_equal(prof, expected_prof)
import numpy as np
from ..._shared.testing import assert_equal, assert_almost_equal
from ..profile import profile_line
def test_45deg_right_downward_interpolated():
    prof = profile_line(image, (2, 2), (8, 8), order=1, mode='constant')
    expected_prof = np.linspace(22, 88, 10)
    assert_almost_equal(prof, expected_prof)
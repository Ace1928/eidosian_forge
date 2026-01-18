import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('n_dims', [2, 3, 5, 7, 9])
@pytest.mark.parametrize('n_pts', [3, 17])
def test_include_ends(self, n_dims, n_pts):
    start, end = _generate_spherical_points(n_dims, 2)
    actual = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, n_pts))
    assert_allclose(actual[0], start)
    assert_allclose(actual[-1], end)
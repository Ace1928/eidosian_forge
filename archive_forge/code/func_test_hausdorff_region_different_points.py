import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.spatial import distance
from skimage._shared._warnings import expected_warnings
from skimage.metrics import hausdorff_distance, hausdorff_pair
@pytest.mark.parametrize('points_a', [(5, 4), (4, 5), (3, 4), (4, 3)])
@pytest.mark.parametrize('points_b', [(6, 4), (2, 6), (2, 4), (4, 0)])
def test_hausdorff_region_different_points(points_a, points_b):
    shape = (7, 7)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True
    dist = np.sqrt(sum(((ca - cb) ** 2 for ca, cb in zip(points_a, points_b))))
    d = distance.cdist([points_a], [points_b])
    dist_modified = max(np.mean(np.min(d, axis=0)), np.mean(np.min(d, axis=1)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a, points_b))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b, method='modified'), dist_modified)
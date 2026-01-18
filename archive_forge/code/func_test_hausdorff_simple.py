import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.spatial import distance
from skimage._shared._warnings import expected_warnings
from skimage.metrics import hausdorff_distance, hausdorff_pair
def test_hausdorff_simple():
    points_a = (3, 0)
    points_b = (6, 0)
    shape = (7, 1)
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
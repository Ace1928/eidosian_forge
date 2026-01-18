import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.spatial import distance
from skimage._shared._warnings import expected_warnings
from skimage.metrics import hausdorff_distance, hausdorff_pair
def test_hausdorff_metrics_match():
    points_a = (3, 0)
    points_b = (6, 0)
    shape = (7, 1)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a, points_b))
    euclidean_distance = distance.euclidean(points_a, points_b)
    assert_almost_equal(euclidean_distance, hausdorff_distance(coords_a, coords_b))
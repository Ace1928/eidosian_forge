import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.spatial import distance
from skimage._shared._warnings import expected_warnings
from skimage.metrics import hausdorff_distance, hausdorff_pair
def test_hausdorff_empty():
    empty = np.zeros((0, 2), dtype=bool)
    non_empty = np.zeros((3, 2), dtype=bool)
    assert hausdorff_distance(empty, non_empty) == 0.0
    assert hausdorff_distance(empty, non_empty, method='modified') == 0.0
    with expected_warnings(['One or both of the images is empty']):
        assert_array_equal(hausdorff_pair(empty, non_empty), [(), ()])
    assert hausdorff_distance(non_empty, empty) == 0.0
    assert hausdorff_distance(non_empty, empty, method='modified') == 0.0
    with expected_warnings(['One or both of the images is empty']):
        assert_array_equal(hausdorff_pair(non_empty, empty), [(), ()])
    assert hausdorff_distance(empty, non_empty) == 0.0
    assert hausdorff_distance(empty, non_empty, method='modified') == 0.0
    with expected_warnings(['One or both of the images is empty']):
        assert_array_equal(hausdorff_pair(empty, non_empty), [(), ()])
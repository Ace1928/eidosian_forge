import numpy as np
from skimage.morphology import max_tree, area_closing, area_opening
from skimage.morphology import max_tree_local_maxima, diameter_opening
from skimage.morphology import diameter_closing
from skimage.util import invert
from skimage._shared.testing import assert_array_equal, TestCase
def test_local_maxima(self):
    """local maxima for various data types"""
    data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13], [13, 15, 40, 40, 18, 18, 18, 60, 60, 15], [14, 16, 40, 40, 19, 19, 19, 60, 60, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [15, 16, 18, 19, 19, 20, 19, 19, 18, 16], [14, 16, 18, 19, 19, 19, 19, 19, 18, 16], [14, 16, 80, 80, 19, 19, 19, 100, 100, 16], [13, 15, 80, 80, 18, 18, 18, 100, 100, 15], [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]], dtype=np.uint8)
    expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint64)
    for dtype in [np.uint8, np.uint64, np.int8, np.int64]:
        test_data = data.astype(dtype)
        out = max_tree_local_maxima(test_data, connectivity=1)
        out_bin = out > 0
        assert_array_equal(expected_result, out_bin)
        assert out.dtype == expected_result.dtype
        assert np.max(out) == 5
        P, S = max_tree(test_data)
        out = max_tree_local_maxima(test_data, parent=P, tree_traverser=S)
        assert_array_equal(expected_result, out_bin)
        assert out.dtype == expected_result.dtype
        assert np.max(out) == 5
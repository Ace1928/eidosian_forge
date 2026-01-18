import numpy as np
from skimage.morphology import max_tree, area_closing, area_opening
from skimage.morphology import max_tree_local_maxima, diameter_opening
from skimage.morphology import diameter_closing
from skimage.util import invert
from skimage._shared.testing import assert_array_equal, TestCase
def test_extrema_float(self):
    """specific tests for float type"""
    data = np.array([[0.1, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14, 0.14, 0.13, 0.11], [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13], [0.13, 0.15, 0.4, 0.4, 0.18, 0.18, 0.18, 0.6, 0.6, 0.15], [0.14, 0.16, 0.4, 0.4, 0.19, 0.19, 0.19, 0.6, 0.6, 0.16], [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16], [0.15, 0.182, 0.18, 0.19, 0.204, 0.2, 0.19, 0.19, 0.18, 0.16], [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19, 0.19, 0.18, 0.16], [0.14, 0.16, 0.8, 0.8, 0.19, 0.19, 0.19, 4.0, 1.0, 0.16], [0.13, 0.15, 0.8, 0.8, 0.18, 0.18, 0.18, 1.0, 1.0, 0.15], [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.15, 0.13]], dtype=np.float32)
    expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    out = max_tree_local_maxima(data, connectivity=1)
    out_bin = out > 0
    assert_array_equal(expected_result, out_bin)
    assert np.max(out) == 6
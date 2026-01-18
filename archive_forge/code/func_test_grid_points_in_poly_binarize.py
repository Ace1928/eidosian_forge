import numpy as np
from skimage.measure import points_in_poly, grid_points_in_poly
from skimage._shared.testing import assert_array_equal
def test_grid_points_in_poly_binarize():
    v = np.array([[0, 0], [5, 0], [5, 5]])
    expected = np.array([[2, 0, 0, 0, 0], [3, 3, 0, 0, 0], [3, 1, 3, 0, 0], [3, 1, 1, 3, 0], [3, 1, 1, 1, 3]])
    assert_array_equal(grid_points_in_poly((5, 5), v, binarize=False), expected)
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
def test_explicit_extent(self):
    expected_x_grid = np.array([[-5.0, 0.0, 5.0, 10.0], [-5.0, 0.0, 5.0, 10.0]])
    expected_y_grid = np.array([[7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10]])
    expected_s_grid = np.array([[2.5, 3.5, 2.5, np.nan], [3.0, 4.0, 3.0, 2.0]])
    extent = (-5, 10, 7.5, 10)
    x_grid, y_grid, s_grid = vec_trans._interpolate_to_grid(4, 2, self.x, self.y, self.s, target_extent=extent)
    assert_array_equal(x_grid, expected_x_grid)
    assert_array_equal(y_grid, expected_y_grid)
    assert_array_almost_equal(s_grid, expected_s_grid)
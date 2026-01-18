import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
from cartopy import geodesic
def test_inverse_broadcast(self):
    repeat_start_pts = np.repeat(self.start_pts[0:1, :], 10, axis=0)
    repeat_end_pts = np.repeat(self.end_pts[0:1, :], 10, axis=0)
    repeat_results = np.repeat(self.inverse_solution[0:1, :], 10, axis=0)
    geod_inv1 = self.geod.inverse(self.start_pts[0], repeat_end_pts)
    geod_inv2 = self.geod.inverse(repeat_start_pts, self.end_pts[0])
    assert_array_almost_equal(geod_inv1, repeat_results, decimal=5)
    assert_array_almost_equal(geod_inv2, repeat_results, decimal=5)
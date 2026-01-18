import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
from cartopy import geodesic
def test_direct_broadcast(self):
    repeat_dists = np.repeat(self.data.dist[0:1], 10, axis=0)
    repeat_start_pts = np.repeat(self.start_pts[0:1, :], 10, axis=0)
    repeat_results = np.repeat(self.direct_solution[0:1, :], 10, axis=0)
    geod_dir1 = self.geod.direct(self.start_pts[0], self.data.start_azi[0], repeat_dists)
    geod_dir2 = self.geod.direct(repeat_start_pts, self.data.start_azi[0], self.data.dist[0])
    assert_array_almost_equal(geod_dir1, repeat_results, decimal=5)
    assert_array_almost_equal(geod_dir2, repeat_results, decimal=5)
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_cyclic_has_cyclic(self):
    """Test detection of cyclic point with cyclic keyword"""
    new_lons = np.deg2rad(self.lon2d)
    new_lats = np.deg2rad(self.lat2d)
    r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
    r_lons = np.concatenate((new_lons, np.full((new_lons.shape[0], 1), np.deg2rad(360))), axis=1)
    r_lats = np.concatenate((new_lats, new_lats[:, -1:]), axis=1)
    c_data, c_lons, c_lats = add_cyclic(r_data, x=r_lons, y=r_lats, cyclic=np.deg2rad(360))
    assert_array_equal(c_data, self.c_data2d)
    assert_array_equal(c_lons, r_lons)
    assert_array_equal(c_lats, r_lats)
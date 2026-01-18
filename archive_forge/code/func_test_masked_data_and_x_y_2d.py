import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_masked_data_and_x_y_2d(self):
    """Test masked data and x"""
    new_data = ma.masked_less(self.data2d, 3)
    new_lon = ma.masked_less(self.lon2d, 2)
    c_data, c_lons, c_lats = add_cyclic(new_data, x=new_lon, y=self.lat2d)
    r_data = ma.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
    r_lons = np.concatenate((self.lon2d, np.full((self.lon2d.shape[0], 1), 360)), axis=1)
    assert_array_equal(c_data, r_data)
    assert_array_equal(c_lons, r_lons)
    assert_array_equal(c_lats, self.c_lat2d)
    assert ma.is_masked(c_data)
    assert ma.is_masked(c_lons)
    assert not ma.is_masked(c_lats)
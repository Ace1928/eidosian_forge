import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_data_and_x_2d(self):
    """Test data and x 2d; no keyword name for x"""
    c_data, c_lons = add_cyclic(self.data2d, self.lon2d)
    assert_array_equal(c_data, self.c_data2d)
    assert_array_equal(c_lons, self.c_lon2d)
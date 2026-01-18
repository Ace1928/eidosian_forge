import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_data_and_x_with_axis_3d(self):
    """Test axis keyword data 4d, x 3d"""
    c_data, c_lons = add_cyclic(self.data4d, x=self.lon3d, axis=1)
    assert_array_equal(c_data, self.c_data4d)
    assert_array_equal(c_lons, self.c_lon3d)
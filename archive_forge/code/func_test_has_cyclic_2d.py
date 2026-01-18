import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_has_cyclic_2d(self):
    """Test detection of cyclic point 2d"""
    c_data, c_lons = add_cyclic(self.c_data2d, x=self.c_lon2d)
    assert_array_equal(c_data, self.c_data2d)
    assert_array_equal(c_lons, self.c_lon2d)
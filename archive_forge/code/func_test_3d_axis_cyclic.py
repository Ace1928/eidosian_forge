import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_3d_axis_cyclic(self):
    """Test 3d with axis and cyclic keywords"""
    new_clons = np.deg2rad(self.c_lon3d)
    new_lons = np.deg2rad(self.lon3d)
    assert has_cyclic(new_clons, axis=1, cyclic=np.deg2rad(360))
    assert not has_cyclic(new_lons, axis=1, cyclic=np.deg2rad(360))
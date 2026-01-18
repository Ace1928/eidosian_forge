import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_invalid_x_size_3d(self):
    """Catch wrong x size 3d"""
    with pytest.raises(ValueError):
        c_data, c_lons = add_cyclic(self.data4d, x=self.lon3d[:, :-1, :], axis=1)
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_invalid_y_size(self):
    """Catch wrong y size 2d"""
    with pytest.raises(ValueError):
        c_data, c_lons, c_lats = add_cyclic(self.data2d, x=self.lon2d, y=self.lat2d[:, 1:])
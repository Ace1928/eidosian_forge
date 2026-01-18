import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_invalid_x_size_1d(self):
    """Catch wrong x size 1d"""
    with pytest.raises(ValueError):
        c_data, c_lons = add_cyclic(self.data2d, x=self.lons[:-1])
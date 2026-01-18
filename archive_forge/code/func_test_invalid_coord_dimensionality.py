import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_invalid_coord_dimensionality(self):
    lons2d = np.repeat(self.lons[np.newaxis], 3, axis=0)
    with pytest.raises(ValueError):
        c_data, c_lons = add_cyclic_point(self.data2d, coord=lons2d)
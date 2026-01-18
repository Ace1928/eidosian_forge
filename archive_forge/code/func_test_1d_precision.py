import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_1d_precision(self):
    """Test 1d with precision keyword"""
    new_clons = np.concatenate((self.lons, np.array([360 + 0.001])))
    assert has_cyclic(new_clons, precision=0.01)
    assert not has_cyclic(new_clons, precision=0.0002)
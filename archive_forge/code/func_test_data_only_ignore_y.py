import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_data_only_ignore_y(self):
    """Test y given but no x"""
    c_data = add_cyclic(self.data2d, y=self.lat2d)
    assert_array_equal(c_data, self.c_data2d)
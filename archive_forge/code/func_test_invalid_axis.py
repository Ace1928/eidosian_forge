import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_invalid_axis(self):
    """Catch wrong axis keyword"""
    with pytest.raises(ValueError):
        add_cyclic(self.data2d, axis=-3)
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest
import shapely.geometry as sgeom
from cartopy import geodesic
def test_inverse_shape(self):
    with pytest.raises(ValueError):
        self.geod.inverse([[0, 1, 2], [0, 1, 2]], [2, 3])
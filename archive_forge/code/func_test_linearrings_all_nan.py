import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linearrings_all_nan():
    coords = np.full((4, 2), np.nan)
    with pytest.raises(shapely.GEOSException):
        shapely.linearrings(coords)
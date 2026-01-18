import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('coords', [[np.nan, np.nan, np.nan, np.nan], [np.nan, 0, 1, 1], [0, np.nan, 1, 1], [0, 0, np.nan, 1], [0, 0, 1, np.nan]])
def test_box_nan(coords):
    assert shapely.box(*coords) is None
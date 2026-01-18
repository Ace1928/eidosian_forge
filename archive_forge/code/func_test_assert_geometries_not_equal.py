import numpy as np
import pytest
import shapely
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('use_array', ['none', 'left', 'right', 'both'])
@pytest.mark.parametrize('geom1,geom2', [(point, line_string), (line_string, line_string_z), (empty_point, empty_polygon), pytest.param(empty_point, empty_point_z, marks=PRE_GEOS_390), pytest.param(empty_line_string, empty_line_string_z, marks=PRE_GEOS_390)])
def test_assert_geometries_not_equal(geom1, geom2, use_array):
    with pytest.raises(AssertionError):
        assert_geometries_equal(*make_array(geom1, geom2, use_array))
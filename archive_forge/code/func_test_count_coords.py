import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
@pytest.mark.parametrize('geoms,count', [([], 0), ([empty], 0), ([point, empty], 1), ([empty, point, empty], 1), ([point, None], 1), ([None, point, None], 1), ([point, point], 2), ([point, point_z], 2), ([line_string, linear_ring], 8), ([polygon], 5), ([polygon_with_hole], 10), ([multi_point, multi_line_string], 4), ([multi_polygon], 10), ([geometry_collection], 3), ([nested_2], 4), ([nested_3], 5)])
def test_count_coords(geoms, count):
    actual = count_coordinates(np.array(geoms, np.object_))
    assert actual == count
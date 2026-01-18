import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('func,sub_geom', [(shapely.multipoints, point), (shapely.multilinestrings, line_string), (shapely.multilinestrings, linear_ring), (shapely.multipolygons, polygon), (shapely.geometrycollections, point), (shapely.geometrycollections, line_string), (shapely.geometrycollections, linear_ring), (shapely.geometrycollections, polygon), (shapely.geometrycollections, multi_point), (shapely.geometrycollections, multi_line_string), (shapely.geometrycollections, multi_polygon), (shapely.geometrycollections, geometry_collection)])
def test_create_collection(func, sub_geom):
    actual = func([sub_geom, sub_geom])
    assert shapely.get_num_geometries(actual) == 2
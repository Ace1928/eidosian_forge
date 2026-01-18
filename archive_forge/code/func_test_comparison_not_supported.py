import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
def test_comparison_not_supported():
    geom1 = Point(1, 1)
    geom2 = Point(2, 2)
    with pytest.raises(TypeError, match='not supported between instances'):
        geom1 > geom2
    with pytest.raises(TypeError, match='not supported between instances'):
        geom1 < geom2
    with pytest.raises(TypeError, match='not supported between instances'):
        geom1 >= geom2
    with pytest.raises(TypeError, match='not supported between instances'):
        geom1 <= geom2
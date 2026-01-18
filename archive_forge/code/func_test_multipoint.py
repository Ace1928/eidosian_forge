import numpy as np
import pytest
from shapely import MultiPoint, Point
from shapely.errors import EmptyPartError
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
def test_multipoint(self):
    geom = MultiPoint([(1.0, 2.0), (3.0, 4.0)])
    assert len(geom.geoms) == 2
    assert dump_coords(geom) == [[(1.0, 2.0)], [(3.0, 4.0)]]
    geom = MultiPoint([Point(1.0, 2.0), Point(3.0, 4.0)])
    assert len(geom.geoms) == 2
    assert dump_coords(geom) == [[(1.0, 2.0)], [(3.0, 4.0)]]
    geom2 = MultiPoint(geom)
    assert len(geom2.geoms) == 2
    assert dump_coords(geom2) == [[(1.0, 2.0)], [(3.0, 4.0)]]
    assert isinstance(geom.geoms[0], Point)
    assert geom.geoms[0].x == 1.0
    assert geom.geoms[0].y == 2.0
    with pytest.raises(IndexError):
        geom.geoms[2]
    assert geom.__geo_interface__ == {'type': 'MultiPoint', 'coordinates': ((1.0, 2.0), (3.0, 4.0))}
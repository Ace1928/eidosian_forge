import numpy as np
import pytest
from shapely import MultiPoint, Point
from shapely.errors import EmptyPartError
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
def test_multipoint_from_numpy(self):
    geom = MultiPoint(np.array([[0.0, 0.0], [1.0, 2.0]]))
    assert isinstance(geom, MultiPoint)
    assert len(geom.geoms) == 2
    assert dump_coords(geom) == [[(0.0, 0.0)], [(1.0, 2.0)]]
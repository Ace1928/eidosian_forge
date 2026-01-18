import numpy as np
import pytest
from shapely import MultiPolygon, Polygon
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
def test_fail_list_of_multipolygons():
    """A list of multipolygons is not a valid multipolygon ctor argument"""
    multi = MultiPolygon([(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)), [((0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25))])])
    with pytest.raises(ValueError):
        MultiPolygon([multi])
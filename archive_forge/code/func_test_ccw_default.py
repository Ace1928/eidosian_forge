import unittest
from shapely import geometry
def test_ccw_default(self):
    b = geometry.box(0, 0, 1, 1)
    assert b.exterior.coords[0] == (1.0, 0.0)
    assert b.exterior.coords[1] == (1.0, 1.0)
import pytest
from shapely import Point
from shapely.errors import ShapelyDeprecationWarning
def test_equals_exact():
    p1 = Point(1.0, 1.0)
    p2 = Point(2.0, 2.0)
    assert not p1.equals(p2)
    assert not p1.equals_exact(p2, 0.001)
    assert p1.equals_exact(p2, 1.42)
import pytest
from shapely import Point
from shapely.errors import ShapelyDeprecationWarning
def test_almost_equals_default():
    p1 = Point(1.0, 1.0)
    p2 = Point(1.0 + 1e-07, 1.0 + 1e-07)
    p3 = Point(1.0 + 1e-06, 1.0 + 1e-06)
    with pytest.warns(ShapelyDeprecationWarning):
        assert p1.almost_equals(p2)
    with pytest.warns(ShapelyDeprecationWarning):
        assert not p1.almost_equals(p3)
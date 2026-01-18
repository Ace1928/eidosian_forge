import pytest
from shapely.geometry import MultiLineString, Polygon, shape
from shapely.geometry.geo import _is_coordinates_empty
@pytest.mark.parametrize('coords', [[], [[]], [[], []], None, [[[]]]])
def test_is_coordinates_empty(coords):
    assert _is_coordinates_empty(coords)
import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
def test_contains_properly():
    polygon = Polygon([(0, 0), (10, 10), (10, -10)])
    line = LineString([(0, 0), (10, 0)])
    assert polygon.contains_properly(line) is False
    assert polygon.contains(line) is True
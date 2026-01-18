from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
def test_binary_empty_result():
    a = LineString([(0, 0), (3, 0), (3, 3), (0, 3)])
    b = LineString([(5, 1), (6, 1)])
    with ignore_invalid(shapely.geos_version < (3, 12, 0)):
        assert shapely.intersection(a, b).is_empty
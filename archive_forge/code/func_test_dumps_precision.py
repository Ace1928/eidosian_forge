from math import pi
import pytest
from shapely.geometry import Point
from shapely.wkt import dump, dumps, load, loads
def test_dumps_precision(some_point):
    assert dumps(some_point, rounding_precision=4) == f'POINT ({pi:.4f} {-pi:.4f})'
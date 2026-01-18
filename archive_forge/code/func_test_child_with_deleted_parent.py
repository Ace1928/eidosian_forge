import numpy as np
import pytest
from shapely import GeometryCollection, LineString, Point, wkt
from shapely.geometry import shape
def test_child_with_deleted_parent():
    a = LineString([(0, 0), (1, 1), (1, 2), (2, 2)])
    b = LineString([(0, 0), (1, 1), (2, 1), (2, 2)])
    collection = a.intersection(b)
    child = collection.geoms[0]
    del collection
    assert child.wkt is not None
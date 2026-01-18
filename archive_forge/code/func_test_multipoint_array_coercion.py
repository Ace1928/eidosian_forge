import numpy as np
import pytest
from shapely import MultiPoint, Point
from shapely.errors import EmptyPartError
from shapely.geometry.base import dump_coords
from shapely.tests.geometry.test_multi import MultiGeometryTestCase
def test_multipoint_array_coercion():
    geom = MultiPoint([(1.0, 2.0), (3.0, 4.0)])
    arr = np.array(geom)
    assert arr.ndim == 0
    assert arr.size == 1
    assert arr.dtype == np.dtype('object')
    assert arr.item() == geom
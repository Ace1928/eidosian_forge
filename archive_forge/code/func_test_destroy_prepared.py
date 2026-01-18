import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_destroy_prepared():
    arr = np.array([shapely.points(1, 1), None, shapely.box(0, 0, 1, 1)])
    shapely.prepare(arr)
    assert arr[0]._geom_prepared != 0
    assert arr[2]._geom_prepared != 0
    shapely.destroy_prepared(arr)
    assert arr[0]._geom_prepared == 0
    assert arr[1] is None
    assert arr[2]._geom_prepared == 0
    shapely.destroy_prepared(arr)
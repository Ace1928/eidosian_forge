import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_2_polygons_with_same_hole():
    actual = shapely.polygons([box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)], [box_tpl(1, 1, 2, 2)])
    assert shapely.area(actual).tolist() == [99.0, 24.0]
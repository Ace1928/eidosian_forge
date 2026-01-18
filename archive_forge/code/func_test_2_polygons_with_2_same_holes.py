import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_2_polygons_with_2_same_holes():
    actual = shapely.polygons([box_tpl(0, 0, 10, 10), box_tpl(0, 0, 5, 5)], [box_tpl(1, 1, 2, 2), box_tpl(3, 3, 4, 4)])
    assert shapely.area(actual).tolist() == [98.0, 23.0]
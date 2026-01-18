import itertools
import math
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from numpy.testing import assert_array_equal
import shapely
from shapely import box, geos_version, LineString, MultiPoint, Point, STRtree
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
import pickle
import sys
from shapely import Point, geos_version
@pytest.mark.parametrize('geometry,expected', [(Point(0.25, 0.25), 0), (Point(0.75, 0.75), 1), (Point(1, 1), 1), ([Point(1, 1), Point(0, 0)], [1, 0]), ([Point(1, 1), Point(0.25, 1)], [1, 1]), ([Point(-10, -10), Point(100, 100)], [0, 9]), (box(0.5, 0.5, 0.75, 0.75), 1), (shapely.buffer(Point(2.5, 2.5), HALF_UNIT_DIAG), 2), (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), 3), (MultiPoint([[5.5, 5], [7, 7]]), 7), (MultiPoint([[5, 7], [7, 5]]), 6)])
def test_nearest_points(tree, geometry, expected):
    assert_array_equal(tree.nearest(geometry), expected)
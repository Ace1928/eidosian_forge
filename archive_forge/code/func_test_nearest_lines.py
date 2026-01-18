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
@pytest.mark.parametrize('geometry,expected', [(Point(0.5, 0.5), 0), (Point(1.5, 0.5), 0), (shapely.box(0.5, 1.5, 1, 2), 1), (shapely.linestrings([[0, 0.5], [1, 2.5]]), 0)])
def test_nearest_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.nearest(geometry), expected)
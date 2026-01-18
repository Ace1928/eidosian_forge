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
def test_geometries_property():
    arr = np.array([point])
    tree = STRtree(arr)
    assert_geometries_equal(arr, tree.geometries)
    arr[0] = shapely.Point(0, 0)
    assert_geometries_equal(point, tree.geometries[0])
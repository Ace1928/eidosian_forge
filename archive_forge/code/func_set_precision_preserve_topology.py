import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('preserve_topology', [False, True])
def set_precision_preserve_topology(preserve_topology):
    with pytest.warns(UserWarning):
        actual = shapely.set_precision(LineString([(0, 0), (0.1, 0.1)]), 1.0, preserve_topology=preserve_topology)
    assert_geometries_equal(shapely.force_2d(actual), LineString())
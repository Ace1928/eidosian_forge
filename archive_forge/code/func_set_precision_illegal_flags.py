import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('flags', [np.array([0, 1]), 4, 'foo'])
def set_precision_illegal_flags(flags):
    with pytest.raises((ValueError, TypeError)):
        shapely.lib.set_precision(line_string, 1.0, flags)
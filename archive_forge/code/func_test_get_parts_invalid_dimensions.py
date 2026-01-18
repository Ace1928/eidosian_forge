import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', ([[None]], [[empty_point]], [[multi_point]], [[multi_point, multi_line_string]]))
def test_get_parts_invalid_dimensions(geom):
    """Only 1D inputs are supported"""
    with pytest.raises(ValueError, match='Array should be one dimensional'):
        shapely.get_parts(geom)
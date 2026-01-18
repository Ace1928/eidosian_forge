import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def geomcol_gdf():
    """Create a Mixed Polygon and LineString For Testing"""
    point = Point(2, 3)
    poly = Polygon([(3, 4), (5, 2), (12, 2), (10, 5), (9, 7.5)])
    coll = GeometryCollection([point, poly])
    gdf = GeoDataFrame([1], geometry=[coll], crs='EPSG:3857')
    return gdf
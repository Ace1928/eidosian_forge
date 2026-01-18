import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def single_rectangle_gdf():
    """Create a single rectangle for clipping."""
    poly_inters = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs='EPSG:3857')
    gdf['attr2'] = 'site-boundary'
    return gdf
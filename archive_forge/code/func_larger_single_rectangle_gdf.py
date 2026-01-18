import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def larger_single_rectangle_gdf():
    """Create a slightly larger rectangle for clipping.
    The smaller single rectangle is used to test the edge case where slivers
    are returned when you clip polygons. This fixture is larger which
    eliminates the slivers in the clip return.
    """
    poly_inters = Polygon([(-5, -5), (-5, 15), (15, 15), (15, -5), (-5, -5)])
    gdf = GeoDataFrame([1], geometry=[poly_inters], crs='EPSG:3857')
    gdf['attr2'] = ['study area']
    return gdf
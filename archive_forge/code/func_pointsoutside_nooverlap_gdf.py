import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def pointsoutside_nooverlap_gdf():
    """Create a point GeoDataFrame. Its points are all outside the single
    rectangle, and its bounds are outside the single rectangle's."""
    pts = np.array([[5, 15], [15, 15], [15, 20]])
    gdf = GeoDataFrame([Point(xy) for xy in pts], columns=['geometry'], crs='EPSG:3857')
    return gdf
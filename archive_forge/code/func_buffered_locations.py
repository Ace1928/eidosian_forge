import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def buffered_locations(point_gdf):
    """Buffer points to create a multi-polygon."""
    buffered_locs = point_gdf
    buffered_locs['geometry'] = buffered_locs.buffer(4)
    buffered_locs['type'] = 'plot'
    return buffered_locs
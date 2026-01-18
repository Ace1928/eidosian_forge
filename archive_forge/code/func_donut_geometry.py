import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def donut_geometry(buffered_locations, single_rectangle_gdf):
    """Make a geometry with a hole in the middle (a donut)."""
    donut = geopandas.overlay(buffered_locations, single_rectangle_gdf, how='symmetric_difference')
    return donut
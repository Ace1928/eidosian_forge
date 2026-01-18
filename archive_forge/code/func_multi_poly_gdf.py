import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.fixture
def multi_poly_gdf(donut_geometry):
    """Create a multi-polygon GeoDataFrame."""
    multi_poly = donut_geometry.unary_union
    out_df = GeoDataFrame(geometry=GeoSeries(multi_poly), crs='EPSG:3857')
    out_df['attr'] = ['pool']
    return out_df
import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_multipoly_keep_geom_type(self, multi_poly_gdf, mask):
    """Test a multi poly object where the return includes a sliver.
        Also the bounds of the object should == the bounds of the clip object
        if they fully overlap (as they do in these fixtures)."""
    clipped = clip(multi_poly_gdf, mask, keep_geom_type=True)
    expected_bounds = mask if _mask_is_list_like_rectangle(mask) else mask.total_bounds
    assert np.array_equal(clipped.total_bounds, expected_bounds)
    assert clipped.geom_type.isin(['Polygon', 'MultiPolygon']).all()
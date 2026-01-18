import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_points_geom_col_rename(self, point_gdf, mask):
    """Test clipping a points GDF with a generic polygon geometry."""
    point_gdf_geom_col_rename = point_gdf.rename_geometry('geometry2')
    clip_pts = clip(point_gdf_geom_col_rename, mask)
    pts = np.array([[2, 2], [3, 4], [9, 8]])
    exp = GeoDataFrame([Point(xy) for xy in pts], columns=['geometry2'], crs='EPSG:3857', geometry='geometry2')
    assert_geodataframe_equal(clip_pts, exp)
import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_lines(self, two_line_gdf, mask):
    """Test what happens when you give the clip_extent a line GDF."""
    clip_line = clip(two_line_gdf, mask)
    assert len(clip_line.geometry) == 2
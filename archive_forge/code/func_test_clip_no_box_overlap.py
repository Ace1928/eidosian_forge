import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_no_box_overlap(self, pointsoutside_nooverlap_gdf, mask):
    """Test clip when intersection is empty and boxes do not overlap."""
    clipped = clip(pointsoutside_nooverlap_gdf, mask)
    assert len(clipped) == 0
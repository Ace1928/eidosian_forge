import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_poly_series(self, buffered_locations, mask):
    """Test clipping a polygon GDF with a generic polygon geometry."""
    clipped_poly = clip(buffered_locations.geometry, mask)
    assert len(clipped_poly) == 3
    assert all(clipped_poly.geom_type == 'Polygon')
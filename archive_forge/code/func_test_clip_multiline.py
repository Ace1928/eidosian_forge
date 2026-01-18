import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
def test_clip_multiline(self, multi_line, mask):
    """Test that clipping a multiline feature with a poly returns expected
        output."""
    clipped = clip(multi_line, mask)
    assert clipped.geom_type[0] == 'MultiLineString'
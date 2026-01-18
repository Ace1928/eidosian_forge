import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.mark.filterwarnings('ignore:All-NaN slice encountered')
@pytest.mark.parametrize('mask', [Polygon(), (np.nan,) * 4, (np.nan, 0, np.nan, 1), GeoSeries([Polygon(), Polygon()], crs='EPSG:3857'), GeoSeries([Polygon(), Polygon()], crs='EPSG:3857').to_frame(), GeoSeries([], crs='EPSG:3857'), GeoSeries([], crs='EPSG:3857').to_frame()])
def test_clip_empty_mask(buffered_locations, mask):
    """Test that clipping with empty mask returns an empty result."""
    clipped = clip(buffered_locations, mask)
    assert_geodataframe_equal(clipped, GeoDataFrame([], columns=['geometry', 'type'], crs='EPSG:3857'), check_index_type=False)
    clipped = clip(buffered_locations.geometry, mask)
    assert_geoseries_equal(clipped, GeoSeries([], crs='EPSG:3857'))
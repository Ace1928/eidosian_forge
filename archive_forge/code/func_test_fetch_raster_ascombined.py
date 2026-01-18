import numpy as np
from numpy.testing import assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.io.srtm
from .test_downloaders import download_to_temp  # noqa: F401 (used as fixture)
@pytest.mark.parametrize('Source', [cartopy.io.srtm.SRTM3Source, cartopy.io.srtm.SRTM1Source], ids=['srtm3', 'srtm1'])
def test_fetch_raster_ascombined(Source):
    source = Source()
    e_img, e_crs, e_extent = source.combined(-1, 50, 2, 1)
    imgs = source.fetch_raster(ccrs.PlateCarree(), (-0.9, 0.1, 50.1, 50.999), None)
    assert len(imgs) == 1
    r_img, r_extent = imgs[0]
    assert e_extent == r_extent
    assert_array_equal(e_img[::-1, :], r_img)
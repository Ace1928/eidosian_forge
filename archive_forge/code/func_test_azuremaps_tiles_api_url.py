import hashlib
import os
import types
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_arr_almost
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
def test_azuremaps_tiles_api_url():
    subscription_key = 'foo'
    tileset_id = 'bar'
    tile = [0, 1, 2]
    exp_url = 'https://atlas.microsoft.com/map/tile?api-version=2.0&tilesetId=bar&x=0&y=1&zoom=2&subscription-key=foo'
    az_maps_sample = cimgt.AzureMapsTiles(subscription_key, tileset_id)
    url_str = az_maps_sample._image_url(tile)
    assert url_str == exp_url
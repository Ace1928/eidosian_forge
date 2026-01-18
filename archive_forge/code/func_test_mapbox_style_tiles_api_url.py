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
def test_mapbox_style_tiles_api_url():
    token = 'foo'
    username = 'baz'
    map_id = 'bar'
    tile = [0, 1, 2]
    exp_url = 'https://api.mapbox.com/styles/v1/baz/bar/tiles/256/2/0/1?access_token=foo'
    mapbox_sample = cimgt.MapboxStyleTiles(token, username, map_id)
    url_str = mapbox_sample._image_url(tile)
    assert url_str == exp_url
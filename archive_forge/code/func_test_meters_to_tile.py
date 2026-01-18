from __future__ import annotations
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import viridis
from datashader.tiles import render_tiles
from datashader.tiles import gen_super_tiles
from datashader.tiles import _get_super_tile_min_max
from datashader.tiles import calculate_zoom_level_stats
from datashader.tiles import MercatorTileDefinition
import numpy as np
import pandas as pd
def test_meters_to_tile():
    full_extent_of_data = (-8243206.93436, 4968192.04221, -8226510.539480001, 4982886.20438)
    xmin, ymin, xmax, ymax = full_extent_of_data
    zoom = 12
    tile_def = MercatorTileDefinition((xmin, xmax), (ymin, ymax), tile_size=256)
    tile = tile_def.meters_to_tile(xmin, ymin, zoom)
    assert tile == (1205, 1540)
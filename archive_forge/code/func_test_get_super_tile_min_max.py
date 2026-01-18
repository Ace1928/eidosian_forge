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
def test_get_super_tile_min_max():
    tile_info = {'level': 0, 'x_range': (-MERCATOR_CONST, MERCATOR_CONST), 'y_range': (-MERCATOR_CONST, MERCATOR_CONST), 'tile_size': 256, 'span': (0, 1000)}
    agg = _get_super_tile_min_max(tile_info, mock_load_data_func, mock_rasterize_func)
    result = [np.nanmin(agg.data), np.nanmax(agg.data)]
    assert isinstance(result, list)
    assert len(result) == 2
    assert_is_numeric(result[0])
    assert_is_numeric(result[1])
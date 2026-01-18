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
def test_calculate_zoom_level_stats_with_fullscan_ranging_strategy():
    full_extent = (-MERCATOR_CONST, -MERCATOR_CONST, MERCATOR_CONST, MERCATOR_CONST)
    level = 0
    color_ranging_strategy = 'fullscan'
    super_tiles, span = calculate_zoom_level_stats(list(gen_super_tiles(full_extent, level)), mock_load_data_func, mock_rasterize_func, color_ranging_strategy=color_ranging_strategy)
    assert isinstance(span, (list, tuple))
    assert len(span) == 2
    assert_is_numeric(span[0])
    assert_is_numeric(span[1])
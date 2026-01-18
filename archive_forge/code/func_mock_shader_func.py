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
def mock_shader_func(agg, span=None):
    img = tf.shade(agg, cmap=viridis, span=span, how='log')
    img = tf.set_background(img, 'black')
    return img
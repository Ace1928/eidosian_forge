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
def mock_load_data_func(x_range, y_range):
    global df
    if df is None:
        xs = np.random.normal(loc=0, scale=500000, size=10000000)
        ys = np.random.normal(loc=0, scale=500000, size=10000000)
        df = pd.DataFrame(dict(x=xs, y=ys))
    return df.loc[df['x'].between(*x_range) & df['y'].between(*y_range)]
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
def assert_is_numeric(value):
    is_int_or_float = isinstance(value, (int, float))
    type_name = type(value).__name__
    is_numpy_int_or_float = 'int' in type_name or 'float' in type_name
    assert any([is_int_or_float, is_numpy_int_or_float])
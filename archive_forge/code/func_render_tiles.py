from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def render_tiles(full_extent, levels, load_data_func, rasterize_func, shader_func, post_render_func, output_path, color_ranging_strategy='fullscan'):
    results = {}
    for level in levels:
        print('calculating statistics for level {}'.format(level))
        super_tiles, span = calculate_zoom_level_stats(list(gen_super_tiles(full_extent, level)), load_data_func, rasterize_func, color_ranging_strategy=color_ranging_strategy)
        print('rendering {} supertiles for zoom level {} with span={}'.format(len(super_tiles), level, span))
        b = db.from_sequence(super_tiles)
        b.map(render_super_tile, span, output_path, shader_func, post_render_func).compute()
        results[level] = dict(success=True, stats=span, supertile_count=len(super_tiles))
    return results
from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def render_super_tile(tile_info, span, output_path, shader_func, post_render_func):
    level = tile_info['level']
    ds_img = shader_func(tile_info['agg'], span=span)
    return create_sub_tiles(ds_img, level, tile_info, output_path, post_render_func)
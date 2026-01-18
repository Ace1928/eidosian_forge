import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
def round_window_to_full_blocks(window, block_shapes, height=0, width=0):
    """Round window to include full expanse of intersecting tiles.

    Parameters
    ----------
    window: Window
        The input window.

    block_shapes : tuple of block shapes
        The input raster's block shape. All bands must have the same
        block/stripe structure

    Returns
    -------
    Window
    """
    if len(set(block_shapes)) != 1:
        raise WindowError('All bands must have the same block/stripe structure')
    window = evaluate(window, height=height, width=width)
    height_shape = block_shapes[0][0]
    width_shape = block_shapes[0][1]
    (row_start, row_stop), (col_start, col_stop) = window.toranges()
    row_min = int(row_start // height_shape) * height_shape
    row_max = int(row_stop // height_shape) * height_shape + (height_shape if row_stop % height_shape != 0 else 0)
    col_min = int(col_start // width_shape) * width_shape
    col_max = int(col_stop // width_shape) * width_shape + (width_shape if col_stop % width_shape != 0 else 0)
    return Window(col_min, row_min, col_max - col_min, row_max - row_min)
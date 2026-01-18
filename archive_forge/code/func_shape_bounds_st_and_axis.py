from datashader.compiler import compile_components
from datashader.utils import Dispatcher
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.glyphs.quadmesh import (
from datashader.utils import apply
import dask
import numpy as np
import xarray as xr
from dask.base import tokenize, compute
from dask.array.overlap import overlap
def shape_bounds_st_and_axis(xr_ds, canvas, glyph):
    if not canvas.x_range or not canvas.y_range:
        x_extents, y_extents = glyph.compute_bounds_dask(xr_ds)
    else:
        x_extents, y_extents = (None, None)
    x_range = canvas.x_range or x_extents
    y_range = canvas.y_range or y_extents
    x_min, x_max, y_min, y_max = bounds = compute(*x_range + y_range)
    x_range, y_range = ((x_min, x_max), (y_min, y_max))
    width = canvas.plot_width
    height = canvas.plot_height
    x_st = canvas.x_axis.compute_scale_and_translate(x_range, width)
    y_st = canvas.y_axis.compute_scale_and_translate(y_range, height)
    st = x_st + y_st
    shape = (height, width)
    x_axis = canvas.x_axis.compute_index(x_st, width)
    y_axis = canvas.y_axis.compute_index(y_st, height)
    axis = dict([(glyph.x_label, x_axis), (glyph.y_label, y_axis)])
    return (shape, bounds, st, axis)
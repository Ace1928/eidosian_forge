from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
@ngjit
@expand_aggs_and_cols
def perform_extend_area_to_line_axis1_ragged(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_inds, x_flat, y0_start_inds, y0_flat, y1_start_inds, y1_flat, *aggs_and_cols):
    nrows = len(x_start_inds)
    x_flat_len = len(x_flat)
    y0_flat_len = len(y0_flat)
    y1_flat_len = len(y1_flat)
    i = 0
    while i < nrows:
        x_start_i = x_start_inds[i]
        x_stop_i = x_start_inds[i + 1] if i < nrows - 1 else x_flat_len
        y0_start_i = y0_start_inds[i]
        y0_stop_i = y0_start_inds[i + 1] if i < nrows - 1 else y0_flat_len
        y1_start_i = y1_start_inds[i]
        y1_stop_i = y1_start_inds[i + 1] if i < nrows - 1 else y1_flat_len
        segment_len = min(x_stop_i - x_start_i, y0_stop_i - y0_start_i, y1_stop_i - y1_start_i)
        j = 0
        while j < segment_len - 1:
            x0 = x_flat[x_start_i + j]
            x1 = x_flat[x_start_i + j + 1]
            y0 = y0_flat[y0_start_i + j]
            y1 = y1_flat[y1_start_i + j]
            y2 = y1_flat[y1_start_i + j + 1]
            y3 = y0_flat[y0_start_i + j + 1]
            trapezoid_start = j == 0 or isnull(x_flat[x_start_i + j - 1]) or isnull(y0_flat[y0_start_i + j - 1]) or isnull(y1_flat[y1_start_i + j] - 1)
            stacked = True
            draw_trapezoid_y(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols)
            j += 1
        i += 1
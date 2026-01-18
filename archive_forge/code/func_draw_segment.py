from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
@ngjit
@expand_aggs_and_cols
def draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols):
    skip = False
    if isnull(x0) or isnull(y0) or isnull(x1) or isnull(y1):
        skip = True
    x0_1, x1_1, y0_1, y1_1, skip, clipped_start, clipped_end = _liang_barsky(xmin, xmax, ymin, ymax, x0, x1, y0, y1, skip)
    if not skip:
        clipped = clipped_start or clipped_end
        segment_start = segment_start or clipped_start
        x0_2, y0_2 = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0_1, y0_1)
        x1_2, y1_2 = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1_1, y1_1)
        if line_width > 0.0:
            if segment_start:
                xm_2 = ym_2 = 0.0
            else:
                xm_2, ym_2 = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xm, ym)
            nx = round((xmax - xmin) * sx)
            ny = round((ymax - ymin) * sy)
            _full_antialias(line_width, overwrite, i, x0_2, x1_2, y0_2, y1_2, segment_start, segment_end, xm_2, ym_2, append, nx, ny, buffer, *aggs_and_cols)
        else:
            _bresenham(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, x0_2, x1_2, y0_2, y1_2, clipped, append, *aggs_and_cols)
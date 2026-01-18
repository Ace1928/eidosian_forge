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
def extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, offsets, outer_offsets, closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols):
    antialias = antialias_stage_2 is not None
    buffer = np.empty(8) if antialias else None
    n_multilines = len(outer_offsets) - 1
    first_pass = True
    for i in range(n_multilines):
        start0 = outer_offsets[i]
        stop0 = outer_offsets[i + 1]
        for j in range(start0, stop0):
            start1 = offsets[j]
            stop1 = offsets[j + 1]
            for k in range(2 * start1, 2 * stop1 - 2, 2):
                x0 = values[k]
                y0 = values[k + 1]
                x1 = values[k + 2]
                y1 = values[k + 3]
                if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
                    continue
                segment_start = k == start1 and (not closed_rings) or (k > start1 and (not np.isfinite(values[k - 2]) or not np.isfinite(values[k - 1])))
                segment_end = not closed_rings and k == stop1 - 4 or (k < stop1 - 4 and (not np.isfinite(values[k + 4]) or not np.isfinite(values[k + 5])))
                draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, 0.0, 0.0, buffer, *aggs_and_cols)
        aa_stage_2_accumulate(aggs_and_accums, first_pass)
        first_pass = False
        aa_stage_2_clear(aggs_and_accums)
    aa_stage_2_copy_back(aggs_and_accums)
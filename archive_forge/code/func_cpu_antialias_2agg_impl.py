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
def cpu_antialias_2agg_impl(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, antialias_stage_2, aggs_and_accums, *aggs_and_cols):
    antialias = antialias_stage_2 is not None
    buffer = np.empty(8) if antialias else None
    ncols = xs.shape[1]
    for i in range(xs.shape[0]):
        for j in range(ncols - 1):
            perform_extend_line(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, buffer, *aggs_and_cols)
        if xs.shape[0] == 1:
            return
        aa_stage_2_accumulate(aggs_and_accums, i == 0)
        if i < xs.shape[0] - 1:
            aa_stage_2_clear(aggs_and_accums)
    aa_stage_2_copy_back(aggs_and_accums)
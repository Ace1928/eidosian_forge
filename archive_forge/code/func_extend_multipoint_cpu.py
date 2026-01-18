from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit
from numba import cuda
@ngjit
@self.expand_aggs_and_cols(append)
def extend_multipoint_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets, eligible_inds, *aggs_and_cols):
    for i in eligible_inds:
        if missing[i] is True:
            continue
        start = offsets[i]
        stop = offsets[i + 1]
        for j in range(start, stop, 2):
            _perform_extend_points(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols)
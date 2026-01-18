from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geometry, antialias_stage_2, *aggs_and_cols):
    coords, offsets, outer_offsets, closed_rings = _process_geometry(geometry)
    n_aggs = len(antialias_stage_2[0])
    aggs_and_accums = tuple(((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs]))
    extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, offsets, outer_offsets, closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols)
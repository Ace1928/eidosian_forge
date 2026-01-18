from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
@ngjit
def clamp_y_indices(ystarti, ystopi, ymaxi):
    """Utility function to compute clamped y-indices"""
    out_of_bounds = ystarti < 0 and ystopi <= 0 or (ystarti > ymaxi and ystopi >= ymaxi)
    clamped_ystarti = max(0, min(ymaxi, ystarti))
    clamped_ystopi = max(-1, min(ymaxi + 1, ystopi))
    return (out_of_bounds, clamped_ystarti, clamped_ystopi)
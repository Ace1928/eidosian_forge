from __future__ import annotations
from math import floor
import numpy as np
from toolz import memoize
from datashader.glyphs.points import _PointLike
from datashader.utils import isreal, ngjit
@ngjit
def map_onto_pixel(vt, bounds, x, y):
    """Map points onto pixel grid.
        """
    sx, tx, sy, ty = vt
    xx = x_mapper(x) * sx + tx - 0.5
    yy = y_mapper(y) * sy + ty - 0.5
    return (xx, yy)
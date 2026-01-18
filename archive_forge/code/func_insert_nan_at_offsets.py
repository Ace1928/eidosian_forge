from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def insert_nan_at_offsets(points: cpy.PointArray, offsets: cpy.OffsetArray) -> cpy.PointArray:
    """Insert NaNs into a point array at locations specified by an offset array.
    """
    check_point_array(points)
    check_offset_array(offsets)
    if len(offsets) <= 2:
        return points
    else:
        nan_spacer = np.array([np.nan, np.nan], dtype=point_dtype)
        return np.insert(points, offsets[1:-1].astype(np.int64), nan_spacer, axis=0)
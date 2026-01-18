from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def codes_from_offsets_and_points(offsets: cpy.OffsetArray, points: cpy.PointArray) -> cpy.CodeArray:
    """Determine codes from offsets and points, using the equality of the start and end points of
    each line to determine if lines are closed or not.
    """
    check_offset_array(offsets)
    check_point_array(points)
    codes = np.full(len(points), LINETO, dtype=code_dtype)
    codes[offsets[:-1]] = MOVETO
    end_offsets = offsets[1:] - 1
    closed = np.all(points[offsets[:-1]] == points[end_offsets], axis=1)
    codes[end_offsets[closed]] = CLOSEPOLY
    return codes
from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def offsets_from_lengths(list_of_points: list[cpy.PointArray]) -> cpy.OffsetArray:
    """Determine offsets from lengths of point arrays.
    """
    if not list_of_points:
        raise ValueError('Empty list passed to offsets_from_lengths')
    return np.cumsum([0] + [len(line) for line in list_of_points], dtype=offset_dtype)
from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def concat_points_or_none(list_of_points_or_none: list[cpy.PointArray | None]) -> cpy.PointArray | None:
    """Concatenate a list of point arrays or None into a single point array or None.
    """
    list_of_points = [points for points in list_of_points_or_none if points is not None]
    if list_of_points:
        return concat_points(list_of_points)
    else:
        return None
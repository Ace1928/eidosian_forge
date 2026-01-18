from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def concat_offsets(list_of_offsets: list[cpy.OffsetArray]) -> cpy.OffsetArray:
    """Concatenate a list of offsets arrays into a single offset array.
    """
    if not list_of_offsets:
        raise ValueError('Empty list passed to concat_offsets')
    n = len(list_of_offsets)
    cumulative = np.cumsum([offsets[-1] for offsets in list_of_offsets], dtype=offset_dtype)
    ret: cpy.OffsetArray = np.concatenate((list_of_offsets[0], *(list_of_offsets[i + 1][1:] + cumulative[i] for i in range(n - 1))), dtype=offset_dtype)
    return ret
from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def outer_offsets_from_list_of_codes(list_of_codes: list[cpy.CodeArray]) -> cpy.OffsetArray:
    """Determine outer offsets from codes using locations of MOVETO codes.
    """
    if not list_of_codes:
        raise ValueError('Empty list passed to outer_offsets_from_list_of_codes')
    return np.cumsum([0] + [np.count_nonzero(codes == MOVETO) for codes in list_of_codes], dtype=offset_dtype)
from __future__ import annotations
from itertools import chain
from typing import TYPE_CHECKING
import numpy as np
from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype
def concat_codes(list_of_codes: list[cpy.CodeArray]) -> cpy.CodeArray:
    """Concatenate a list of codes arrays into a single code array.
    """
    if not list_of_codes:
        raise ValueError('Empty list passed to concat_codes')
    return np.concatenate(list_of_codes, dtype=code_dtype)
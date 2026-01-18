from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def find_in_coord_list(coord_list, coord, atol: float=1e-08):
    """Find the indices of matches of a particular coord in a coord_list.

    Args:
        coord_list: List of coords to test
        coord: Specific coordinates
        atol: Absolute tolerance. Defaults to 1e-8. Accepts both scalar and
            array.

    Returns:
        Indices of matches, e.g., [0, 1, 2, 3]. Empty list if not found.
    """
    if len(coord_list) == 0:
        return []
    diff = np.array(coord_list) - np.array(coord)[None, :]
    return np.where(np.all(np.abs(diff) < atol, axis=1))[0]
from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def find_in_coord_list_pbc(fcoord_list, fcoord, atol: float=1e-08, pbc: PbcLike=(True, True, True)) -> np.ndarray:
    """Get the indices of all points in a fractional coord list that are
    equal to a fractional coord (with a tolerance), taking into account
    periodic boundary conditions.

    Args:
        fcoord_list: List of fractional coords
        fcoord: A specific fractional coord to test.
        atol: Absolute tolerance. Defaults to 1e-8.
        pbc: a tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        Indices of matches, e.g., [0, 1, 2, 3]. Empty list if not found.
    """
    if len(fcoord_list) == 0:
        return []
    frac_coords = np.tile(fcoord, (len(fcoord_list), 1))
    frac_dist = fcoord_list - frac_coords
    frac_dist[:, pbc] -= np.round(frac_dist)[:, pbc]
    return np.where(np.all(np.abs(frac_dist) < atol, axis=1))[0]
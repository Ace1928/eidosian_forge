from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def is_coord_subset_pbc(subset, superset, atol: float=1e-08, mask=None, pbc: PbcLike=(True, True, True)) -> bool:
    """Tests if all fractional coords in subset are contained in superset.

    Args:
        subset (list): List of fractional coords to test
        superset (list): List of fractional coords to test against
        atol (float or size 3 array): Tolerance for matching
        mask (boolean array): Mask of matches that are not allowed.
            i.e. if mask[1,2] is True, then subset[1] cannot be matched
            to superset[2]
        pbc (tuple): a tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        bool: True if all of subset is in superset.
    """
    c1 = np.array(subset, dtype=np.float64)
    c2 = np.array(superset, dtype=np.float64)
    m = np.array(mask, dtype=int) if mask is not None else np.zeros((len(subset), len(superset)), dtype=int)
    atol = np.zeros(3, dtype=np.float64) + atol
    return coord_cython.is_coord_subset_pbc(c1, c2, atol, m, pbc)
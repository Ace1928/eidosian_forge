from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def in_coord_list_pbc(fcoord_list, fcoord, atol: float=1e-08, pbc: PbcLike=(True, True, True)) -> bool:
    """Tests if a particular fractional coord is within a fractional coord_list.

    Args:
        fcoord_list: List of fractional coords to test
        fcoord: A specific fractional coord to test.
        atol: Absolute tolerance. Defaults to 1e-8.
        pbc: a tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        bool: True if coord is in the coord list.
    """
    return len(find_in_coord_list_pbc(fcoord_list, fcoord, atol=atol, pbc=pbc)) > 0
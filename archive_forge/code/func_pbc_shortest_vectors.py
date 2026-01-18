from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def pbc_shortest_vectors(lattice, fcoords1, fcoords2, mask=None, return_d2: bool=False):
    """Returns the shortest vectors between two lists of coordinates taking into
    account periodic boundary conditions and the lattice.

    Args:
        lattice: lattice to use
        fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6, 0.7]
            or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
            coord or any array of coords.
        fcoords2: Second set of fractional coordinates.
        mask (boolean array): Mask of matches that are not allowed.
            i.e. if mask[1,2] is True, then subset[1] cannot be matched
            to superset[2]
        return_d2 (bool): whether to also return the squared distances

    Returns:
        array of displacement vectors from fcoords1 to fcoords2
        first index is fcoords1 index, second is fcoords2 index
    """
    return coord_cython.pbc_shortest_vectors(lattice, fcoords1, fcoords2, mask, return_d2)
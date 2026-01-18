from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def pbc_diff(fcoords1: ArrayLike, fcoords2: ArrayLike, pbc: PbcLike=(True, True, True)):
    """Returns the 'fractional distance' between two coordinates taking into
    account periodic boundary conditions.

    Args:
        fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6,
            0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
            coord or any array of coords.
        fcoords2: Second set of fractional coordinates.
        pbc: a tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        Fractional distance. Each coordinate must have the property that
        abs(a) <= 0.5. Examples:
        pbc_diff([0.1, 0.1, 0.1], [0.3, 0.5, 0.9]) = [-0.2, -0.4, 0.2]
        pbc_diff([0.9, 0.1, 1.01], [0.3, 0.5, 0.9]) = [-0.4, -0.4, 0.11]
    """
    fdist = np.subtract(fcoords1, fcoords2)
    return fdist - np.round(fdist) * pbc
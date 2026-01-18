from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def all_distances(coords1: ArrayLike, coords2: ArrayLike) -> np.ndarray:
    """Returns the distances between two lists of coordinates.

    Args:
        coords1: First set of Cartesian coordinates.
        coords2: Second set of Cartesian coordinates.

    Returns:
        2d array of Cartesian distances. E.g the distance between
        coords1[i] and coords2[j] is distances[i,j]
    """
    c1 = np.array(coords1)
    c2 = np.array(coords2)
    z = (c1[:, None, :] - c2[None, :, :]) ** 2
    return np.sum(z, axis=-1) ** 0.5
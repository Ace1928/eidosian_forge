from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def barycentric_coords(coords, simplex):
    """Converts a list of coordinates to barycentric coordinates, given a
    simplex with d+1 points. Only works for d >= 2.

    Args:
        coords: list of n coords to transform, shape should be (n,d)
        simplex: list of coordinates that form the simplex, shape should be
            (d+1, d)

    Returns:
        a list of barycentric coordinates (even if the original input was 1d)
    """
    coords = np.atleast_2d(coords)
    t = np.transpose(simplex[:-1, :]) - np.transpose(simplex[-1, :])[:, None]
    all_but_one = np.transpose(np.linalg.solve(t, np.transpose(coords - simplex[-1])))
    last_coord = 1 - np.sum(all_but_one, axis=-1)[:, None]
    return np.append(all_but_one, last_coord, axis=-1)
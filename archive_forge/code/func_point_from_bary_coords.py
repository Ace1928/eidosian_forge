from __future__ import annotations
import itertools
import math
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util import coord_cython
def point_from_bary_coords(self, bary_coords: ArrayLike):
    """
        Args:
            bary_coords (ArrayLike): Barycentric coordinates (d+1, d).

        Returns:
            np.array: Point in the simplex.
        """
    try:
        return np.dot(bary_coords, self._aug[:, :-1])
    except AttributeError as exc:
        raise ValueError('Simplex is not full-dimensional') from exc
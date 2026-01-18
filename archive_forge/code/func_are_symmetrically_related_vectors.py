from __future__ import annotations
import re
import string
import typing
import warnings
from math import cos, pi, sin, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.electronic_structure.core import Magmom
from pymatgen.util.due import Doi, due
from pymatgen.util.string import transformation_to_string
def are_symmetrically_related_vectors(self, from_a: ArrayLike, to_a: ArrayLike, r_a: ArrayLike, from_b: ArrayLike, to_b: ArrayLike, r_b: ArrayLike, tol: float=0.001) -> tuple[bool, bool]:
    """Checks if two vectors, or rather two vectors that connect two points
        each are symmetrically related. r_a and r_b give the change of unit
        cells. Two vectors are also considered symmetrically equivalent if starting
        and end point are exchanged.

        Args:
            from_a (3x1 array): Starting point of the first vector.
            to_a (3x1 array): Ending point of the first vector.
            from_b (3x1 array): Starting point of the second vector.
            to_b (3x1 array): Ending point of the second vector.
            r_a (3x1 array): Change of unit cell of the first vector.
            r_b (3x1 array): Change of unit cell of the second vector.
            tol (float): Absolute tolerance for checking distance.

        Returns:
            tuple[bool, bool]: First bool indicates if the vectors are related,
                the second if the vectors are related but the starting and end point
                are exchanged.
        """
    from_c = self.operate(from_a)
    to_c = self.operate(to_a)
    floored = np.floor([from_c, to_c])
    is_too_close = np.abs([from_c, to_c] - floored) > 1 - tol
    floored[is_too_close] += 1
    r_c = self.apply_rotation_only(r_a) - floored[0] + floored[1]
    from_c = from_c % 1
    to_c = to_c % 1
    if np.allclose(from_b, from_c, atol=tol) and np.allclose(to_b, to_c) and np.allclose(r_b, r_c, atol=tol):
        return (True, False)
    if np.allclose(to_b, from_c, atol=tol) and np.allclose(from_b, to_c) and np.allclose(r_b, -r_c, atol=tol):
        return (True, True)
    return (False, False)
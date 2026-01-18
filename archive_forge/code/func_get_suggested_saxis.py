from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@staticmethod
def get_suggested_saxis(magmoms):
    """This method returns a suggested spin axis for a set of magmoms,
        taking the largest magnetic moment as the reference. For calculations
        with collinear spins, this would give a sensible saxis for a ncl
        calculation.

        Args:
            magmoms: list of magmoms (Magmoms, scalars or vectors)

        Returns:
            np.ndarray of length 3
        """
    magmoms = [Magmom(magmom) for magmom in magmoms]
    magmoms = [magmom for magmom in magmoms if abs(magmom)]
    magmoms.sort(reverse=True)
    if len(magmoms) > 0:
        return magmoms[0].get_00t_magmom_with_xyz_saxis().saxis
    return np.array([0, 0, 1], dtype='d')
from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@staticmethod
def get_consistent_set_and_saxis(magmoms, saxis=None):
    """Method to ensure a list of magmoms use the same spin axis.
        Returns a tuple of a list of Magmoms and their global spin axis.

        Args:
            magmoms: list of magmoms (Magmoms, scalars or vectors)
            saxis: can provide a specific global spin axis

        Returns:
            tuple[list[Magmom], np.ndarray]: (list of Magmoms, global spin axis)
        """
    magmoms = [Magmom(magmom) for magmom in magmoms]
    saxis = Magmom.get_suggested_saxis(magmoms) if saxis is None else saxis / np.linalg.norm(saxis)
    magmoms = [magmom.get_moment(saxis=saxis) for magmom in magmoms]
    return (magmoms, saxis)
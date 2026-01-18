from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@staticmethod
def have_consistent_saxis(magmoms) -> bool:
    """This method checks that all Magmom objects in a list have a
        consistent spin quantization axis. To write MAGMOM tags to a
        VASP INCAR, a global SAXIS value for all magmoms has to be used.
        If saxis are inconsistent, can create consistent set with:
        Magmom.get_consistent_set(magmoms).

        Args:
            magmoms: list of magmoms (Magmoms, scalars or vectors)

        Returns:
            bool
        """
    magmoms = [Magmom(magmom) for magmom in magmoms]
    ref_saxis = magmoms[0].saxis
    match_ref = [magmom.saxis == ref_saxis for magmom in magmoms]
    if np.all(match_ref):
        return True
    return False
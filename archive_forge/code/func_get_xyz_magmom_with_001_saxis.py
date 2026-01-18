from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
def get_xyz_magmom_with_001_saxis(self):
    """Returns a Magmom in the default setting of saxis = [0, 0, 1] and
        the magnetic moment rotated as required.

        Returns:
            Magmom
        """
    return Magmom(self.get_moment())
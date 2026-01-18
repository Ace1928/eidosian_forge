from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
def get_moment_relative_to_crystal_axes(self, lattice):
    """If scalar magmoms, moments will be given arbitrarily along z.
        Used for writing moments to magCIF file.

        Args:
            lattice: Lattice

        Returns:
            vector as list of floats
        """
    unit_m = lattice.matrix / np.linalg.norm(lattice.matrix, axis=1)[:, None]
    moment = np.matmul(self.global_moment, np.linalg.inv(unit_m))
    moment[np.abs(moment) < 1e-08] = 0
    return moment
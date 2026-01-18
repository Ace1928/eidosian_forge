from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
@classmethod
def from_moment_relative_to_crystal_axes(cls, moment: list[float], lattice: Lattice) -> Self:
    """Obtaining a Magmom object from a magnetic moment provided
        relative to crystal axes.

        Used for obtaining moments from magCIF file.

        Args:
            moment: list of floats specifying vector magmom
            lattice: Lattice

        Returns:
            Magmom
        """
    unit_m = lattice.matrix / np.linalg.norm(lattice.matrix, axis=1)[:, None]
    moment = np.matmul(list(moment), unit_m)
    moment[np.abs(moment) < 1e-08] = 0
    return cls(moment)
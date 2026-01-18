from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def get_nac_eigendisplacements_along_dir(self, direction) -> np.ndarray | None:
    """Returns the nac_eigendisplacements for the given direction (not necessarily a versor).
        None if the direction is not present or nac_eigendisplacements has not been calculated.

        Args:
            direction: the direction as a list of 3 elements

        Returns:
            the eigendisplacements as a numpy array of complex numbers with shape
            (3*len(structure), len(structure), 3). None if not found.
        """
    versor = [idx / np.linalg.norm(direction) for idx in direction]
    for dist, eigen_disp in self.nac_eigendisplacements:
        if np.allclose(versor, dist):
            return eigen_disp
    return None
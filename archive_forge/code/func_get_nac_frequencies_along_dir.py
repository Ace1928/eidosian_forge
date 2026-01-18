from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def get_nac_frequencies_along_dir(self, direction: Sequence) -> np.ndarray | None:
    """Returns the nac_frequencies for the given direction (not necessarily a versor).
        None if the direction is not present or nac_frequencies has not been calculated.

        Args:
            direction: the direction as a list of 3 elements

        Returns:
            the frequencies as a numpy array o(3*len(structure), len(qpoints)).
            None if not found.
        """
    versor = [idx / np.linalg.norm(direction) for idx in direction]
    for dist, freq in self.nac_frequencies:
        if np.allclose(versor, dist):
            return freq
    return None
from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def eigenvectors_from_displacements(disp: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Calculate the eigenvectors from the atomic displacements."""
    return np.einsum('nax,a->nax', disp, masses ** 0.5)
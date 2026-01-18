from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def get_gamma_point(self) -> Kpoint | None:
    """Returns the Gamma q-point as a Kpoint object (or None if not found)."""
    for q_point in self.qpoints:
        if np.allclose(q_point.frac_coords, (0, 0, 0)):
            return q_point
    return None
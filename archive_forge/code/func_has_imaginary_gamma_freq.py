from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
def has_imaginary_gamma_freq(self, tol: float=0.01) -> bool:
    """Checks if there are imaginary modes at the gamma point and all close points.

        Args:
            tol: Tolerance for determining if a frequency is imaginary. Defaults to 0.01.
        """
    close_points = [q_pt for q_pt in self.qpoints if np.linalg.norm(q_pt.frac_coords) < tol]
    for qpoint in close_points:
        idx = self.qpoints.index(qpoint)
        if any((freq < -tol for freq in self.bands[:, idx])):
            return True
    return False
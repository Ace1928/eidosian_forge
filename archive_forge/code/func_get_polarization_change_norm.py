from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def get_polarization_change_norm(self, convert_to_muC_per_cm2=True, all_in_polar=True):
    """
        Get magnitude of difference between nonpolar and polar same branch
        polarization.
        """
    polar = self.structures[-1]
    a, b, c = polar.lattice.matrix
    a, b, c = (a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c))
    P = self.get_polarization_change(convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar).ravel()
    return np.linalg.norm(a * P[0] + b * P[1] + c * P[2])
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def get_lattice_quanta(self, convert_to_muC_per_cm2=True, all_in_polar=True):
    """
        Returns the dipole / polarization quanta along a, b, and c for
        all structures.
        """
    lattices = [s.lattice for s in self.structures]
    volumes = np.array([struct.volume for struct in self.structures])
    n_structs = len(self.structures)
    e_to_muC = -1.6021766e-13
    cm2_to_A2 = 1e+16
    units = 1 / np.array(volumes)
    units *= e_to_muC * cm2_to_A2
    if convert_to_muC_per_cm2 and (not all_in_polar):
        for idx in range(n_structs):
            lattice = lattices[idx]
            lattices[idx] = Lattice.from_parameters(*np.array(lattice.lengths) * units.ravel()[idx], *lattice.angles)
    elif convert_to_muC_per_cm2 and all_in_polar:
        for idx in range(n_structs):
            lattice = lattices[-1]
            lattices[idx] = Lattice.from_parameters(*np.array(lattice.lengths) * units.ravel()[-1], *lattice.angles)
    return np.array([np.array(latt.lengths) for latt in lattices])
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def get_total_ionic_dipole(structure, zval_dict):
    """
    Get the total ionic dipole moment for a structure.

    structure: pymatgen Structure
    zval_dict: specie, zval dictionary pairs
    center (np.array with shape [3,1]) : dipole center used by VASP
    tiny (float) : tolerance for determining boundary of calculation.
    """
    tot_ionic = []
    for site in structure:
        zval = zval_dict[str(site.specie)]
        tot_ionic.append(calc_ionic(site, structure, zval))
    return np.sum(tot_ionic, axis=0)
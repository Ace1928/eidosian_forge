import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_spin_direction(magmoms):
    """
    From atoms.magmom, returns the spin direction of phi and theta
    """
    if np.array(magmoms).dtype == float or np.array(magmoms).dtype is np.float64:
        return []
    else:
        magmoms = np.array(magmoms)
        return magmoms / np.linalg.norm(magmoms, axis=1)
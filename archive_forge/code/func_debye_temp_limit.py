from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.dev import requires
from monty.json import MSONable
from scipy.interpolate import UnivariateSpline
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import amu_to_kg
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
@property
def debye_temp_limit(self) -> float:
    """Debye temperature in K. Adapted from apipy."""
    f_mesh = self.tdos.frequency_points * const.tera
    dos = self.tdos.dos
    i_a = UnivariateSpline(f_mesh, dos * f_mesh ** 2, s=0).integral(f_mesh[0], f_mesh[-1])
    i_b = UnivariateSpline(f_mesh, dos, s=0).integral(f_mesh[0], f_mesh[-1])
    integrals = i_a / i_b
    return np.sqrt(5 / 3 * integrals) / const.value('Boltzmann constant in Hz/K')
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
def debye_temp_phonopy(self, freq_max_fit=None) -> float:
    """Get Debye temperature in K as implemented in phonopy.

        Args:
            freq_max_fit: Maximum frequency to include for fitting.
                          Defaults to include first quartile of frequencies.

        Returns:
            Debye temperature in K.
        """
    assert self.structure is not None, 'Structure is not defined.'
    t = self.tdos
    t.set_Debye_frequency(num_atoms=len(self.structure), freq_max_fit=freq_max_fit)
    f_d = t.get_Debye_frequency()
    return const.value('Planck constant') * f_d * const.tera / const.value('Boltzmann constant')
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
def internal_energy(self, temp: float | None=None, structure: Structure | None=None, **kwargs) -> float:
    """Phonon contribution to the internal energy at temperature T obtained from the integration of the DOS.
        Only positive frequencies will be used.
        Result in J/mol-c. A mol-c is the abbreviation of a mole-cell, that is, the number
        of Avogadro times the atoms in a unit cell. To compare with experimental data the result
        should be divided by the number of unit formulas in the cell. If the structure is provided
        the division is performed internally and the result is in J/mol.

        Args:
            temp: a temperature in K
            structure: the structure of the system. If not None it will be used to determine the number of
                formula units
            **kwargs: allows passing in deprecated t parameter for temp

        Returns:
            float: Phonon contribution to the internal energy
        """
    temp = kwargs.get('t', temp)
    if temp == 0:
        return self.zero_point_energy(structure=structure)
    freqs = self._positive_frequencies
    dens = self._positive_densities
    wd2kt = freqs / (2 * BOLTZ_THZ_PER_K * temp)
    e_phonon = np.trapz(freqs * 1 / np.tanh(wd2kt) * dens, x=freqs) / 2
    e_phonon *= THZ_TO_J * const.Avogadro
    if structure:
        formula_units = structure.composition.num_atoms / structure.composition.reduced_composition.num_atoms
        e_phonon /= formula_units
    return e_phonon
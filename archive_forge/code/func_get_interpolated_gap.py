from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
def get_interpolated_gap(self, tol: float=0.001, abs_tol: bool=False, spin: Spin | None=None) -> tuple[float, float, float]:
    """Expects a DOS object and finds the gap.

        Args:
            tol: tolerance in occupations for determining the gap
            abs_tol: Set to True for an absolute tolerance and False for a
                relative one.
            spin: Possible values are None - finds the gap in the summed
                densities, Up - finds the gap in the up spin channel,
                Down - finds the gap in the down spin channel.

        Returns:
            tuple[float, float, float]: Energies in eV corresponding to the band gap, cbm and vbm.
        """
    tdos = self.get_densities(spin)
    if not abs_tol:
        tol = tol * tdos.sum() / tdos.shape[0]
    energies = self.energies
    below_fermi = [i for i in range(len(energies)) if energies[i] < self.efermi and tdos[i] > tol]
    above_fermi = [i for i in range(len(energies)) if energies[i] > self.efermi and tdos[i] > tol]
    vbm_start = max(below_fermi)
    cbm_start = min(above_fermi)
    if vbm_start == cbm_start:
        return (0.0, self.efermi, self.efermi)
    terminal_dens = tdos[vbm_start:vbm_start + 2][::-1]
    terminal_energies = energies[vbm_start:vbm_start + 2][::-1]
    start = get_linear_interpolated_value(terminal_dens, terminal_energies, tol)
    terminal_dens = tdos[cbm_start - 1:cbm_start + 1]
    terminal_energies = energies[cbm_start - 1:cbm_start + 1]
    end = get_linear_interpolated_value(terminal_dens, terminal_energies, tol)
    return (end - start, end, start)
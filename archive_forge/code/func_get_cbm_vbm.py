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
def get_cbm_vbm(self, tol: float=0.001, abs_tol: bool=False, spin: Spin | None=None) -> tuple[float, float]:
    """Expects a DOS object and finds the cbm and vbm.

        Args:
            tol: tolerance in occupations for determining the gap
            abs_tol: An absolute tolerance (True) and a relative one (False)
            spin: Possible values are None - finds the gap in the summed
                densities, Up - finds the gap in the up spin channel,
                Down - finds the gap in the down spin channel.

        Returns:
            tuple[float, float]: Energies in eV corresponding to the cbm and vbm.
        """
    tdos = self.get_densities(spin)
    if not abs_tol:
        tol = tol * tdos.sum() / tdos.shape[0]
    i_fermi = 0
    while self.energies[i_fermi] <= self.efermi:
        i_fermi += 1
    i_gap_start = i_fermi
    while i_gap_start - 1 >= 0 and tdos[i_gap_start - 1] <= tol:
        i_gap_start -= 1
    i_gap_end = i_gap_start
    while i_gap_end < tdos.shape[0] and tdos[i_gap_end] <= tol:
        i_gap_end += 1
    i_gap_end -= 1
    return (self.energies[i_gap_end], self.energies[i_gap_start])
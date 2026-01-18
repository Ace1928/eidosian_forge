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
@property
def spin_polarization(self) -> float | None:
    """Calculates spin polarization at Fermi level. If the
        calculation is not spin-polarized, None will be returned.

        See Sanvito et al., doi: 10.1126/sciadv.1602241 for an example usage.

        Returns:
            float: spin polarization in range [0, 1], will also return NaN if spin
                polarization ill-defined (e.g. for insulator).
        """
    n_F = self.get_interpolated_value(self.efermi)
    n_F_up = n_F[Spin.up]
    if Spin.down not in n_F:
        return None
    n_F_down = n_F[Spin.down]
    if n_F_up + n_F_down == 0:
        return float('NaN')
    spin_polarization = (n_F_up - n_F_down) / (n_F_up + n_F_down)
    return abs(spin_polarization)
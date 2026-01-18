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
def get_gap(self, tol: float=0.001, abs_tol: bool=False, spin: Spin | None=None):
    """Expects a DOS object and finds the gap.

        Args:
            tol: tolerance in occupations for determining the gap
            abs_tol: An absolute tolerance (True) and a relative one (False)
            spin: Possible values are None - finds the gap in the summed
                densities, Up - finds the gap in the up spin channel,
                Down - finds the gap in the down spin channel.

        Returns:
            gap in eV
        """
    cbm, vbm = self.get_cbm_vbm(tol, abs_tol, spin)
    return max(cbm - vbm, 0.0)
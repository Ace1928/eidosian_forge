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
def get_normalized(self) -> CompleteDos:
    """Returns a normalized version of the CompleteDos."""
    if self.norm_vol is not None:
        return self
    return CompleteDos(structure=self.structure, total_dos=self, pdoss=self.pdos, normalize=True)
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
@lazy_property
def ind_zero_freq(self) -> int:
    """Index of the first point for which the frequencies are >= 0."""
    ind = np.searchsorted(self.frequencies, 0)
    if ind >= len(self.frequencies):
        raise ValueError('No positive frequencies found')
    return ind
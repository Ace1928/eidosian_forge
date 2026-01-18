from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
def get_direct_band_gap_dict(self):
    """Returns a dictionary of information about the direct
        band gap.

        Returns:
            a dictionary of the band gaps indexed by spin
            along with their band indices and k-point index
        """
    if self.is_metal():
        raise ValueError('get_direct_band_gap_dict should only be used with non-metals')
    direct_gap_dict = {}
    for spin, v in self.bands.items():
        above = v[np.all(v > self.efermi, axis=1)]
        min_above = np.min(above, axis=0)
        below = v[np.all(v < self.efermi, axis=1)]
        max_below = np.max(below, axis=0)
        diff = min_above - max_below
        kpoint_index = np.argmin(diff)
        band_indices = [np.argmax(below[:, kpoint_index]), np.argmin(above[:, kpoint_index]) + len(below)]
        direct_gap_dict[spin] = {'value': diff[kpoint_index], 'kpoint_index': kpoint_index, 'band_indices': band_indices}
    return direct_gap_dict
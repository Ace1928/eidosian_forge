from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
def get_site_energy(self, site_index):
    """Compute the energy for a single site in the structure.

        Args:
            site_index (int): Index of site

        Returns:
            float: Energy of that site
        """
    if not self._initialized:
        self._calc_ewald_terms()
        self._initialized = True
    if self._charged:
        warn('Per atom energies for charged structures not supported in EwaldSummation')
    return np.sum(self._recip[:, site_index]) + np.sum(self._real[:, site_index]) + self._point[site_index]
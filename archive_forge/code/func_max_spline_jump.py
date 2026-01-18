from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def max_spline_jump(self):
    """Get maximum difference between spline and energy trend."""
    sp = self.spline()
    return max(self.energies - sp(range(len(self.energies))))
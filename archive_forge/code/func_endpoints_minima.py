from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import UnivariateSpline
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
def endpoints_minima(self, slope_cutoff=0.005):
    """Test if spline endpoints are at minima for a given slope cutoff."""
    energies = self.energies
    try:
        sp = self.spline()
    except Exception:
        print('Energy spline failed.')
        return None
    der = sp.derivative()
    der_energies = der(range(len(energies)))
    return {'polar': abs(der_energies[-1]) <= slope_cutoff, 'nonpolar': abs(der_energies[0]) <= slope_cutoff}
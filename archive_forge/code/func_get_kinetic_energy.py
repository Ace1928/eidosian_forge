import copy
import numbers
from math import cos, sin, pi
import numpy as np
import ase.units as units
from ase.atom import Atom
from ase.cell import Cell
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.data import atomic_masses, atomic_masses_common
from ase.geometry import (wrap_positions, find_mic, get_angles, get_distances,
from ase.symbols import Symbols, symbols2numbers
from ase.utils import deprecated
def get_kinetic_energy(self):
    """Get the kinetic energy."""
    momenta = self.arrays.get('momenta')
    if momenta is None:
        return 0.0
    return 0.5 * np.vdot(momenta, self.get_velocities())
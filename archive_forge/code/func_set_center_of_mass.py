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
def set_center_of_mass(self, com, scaled=False):
    """Set the center of mass.

        If scaled=True the center of mass is expected in scaled coordinates.
        Constraints are considered for scaled=False.
        """
    old_com = self.get_center_of_mass(scaled=scaled)
    difference = old_com - com
    if scaled:
        self.set_scaled_positions(self.get_scaled_positions() + difference)
    else:
        self.set_positions(self.get_positions() + difference)
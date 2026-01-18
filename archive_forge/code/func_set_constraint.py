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
def set_constraint(self, constraint=None):
    """Apply one or more constrains.

        The *constraint* argument must be one constraint object or a
        list of constraint objects."""
    if constraint is None:
        self._constraints = []
    elif isinstance(constraint, list):
        self._constraints = constraint
    elif isinstance(constraint, tuple):
        self._constraints = list(constraint)
    else:
        self._constraints = [constraint]
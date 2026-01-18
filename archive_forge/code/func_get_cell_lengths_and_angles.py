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
@deprecated('Please use atoms.cell.cellpar() instead')
def get_cell_lengths_and_angles(self):
    """Get unit cell parameters. Sequence of 6 numbers.

        First three are unit cell vector lengths and second three
        are angles between them::

            [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]

        in degrees.
        """
    return self.cell.cellpar()
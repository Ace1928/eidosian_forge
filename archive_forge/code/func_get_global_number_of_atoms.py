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
def get_global_number_of_atoms(self):
    """Returns the global number of atoms in a distributed-atoms parallel
        simulation.

        DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING!

        Equivalent to len(atoms) in the standard ASE Atoms class.  You should
        normally use len(atoms) instead.  This function's only purpose is to
        make compatibility between ASE and Asap easier to maintain by having a
        few places in ASE use this function instead.  It is typically only
        when counting the global number of degrees of freedom or in similar
        situations.
        """
    return len(self)
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
def get_moments_of_inertia(self, vectors=False):
    """Get the moments of inertia along the principal axes.

        The three principal moments of inertia are computed from the
        eigenvalues of the symmetric inertial tensor. Periodic boundary
        conditions are ignored. Units of the moments of inertia are
        amu*angstrom**2.
        """
    com = self.get_center_of_mass()
    positions = self.get_positions()
    positions -= com
    masses = self.get_masses()
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(self)):
        x, y, z = positions[i]
        m = masses[i]
        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z
    I = np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]])
    evals, evecs = np.linalg.eigh(I)
    if vectors:
        return (evals, evecs.transpose())
    else:
        return evals
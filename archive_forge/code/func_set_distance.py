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
def set_distance(self, a0, a1, distance, fix=0.5, mic=False, mask=None, indices=None, add=False, factor=False):
    """Set the distance between two atoms.

        Set the distance between atoms *a0* and *a1* to *distance*.
        By default, the center of the two atoms will be fixed.  Use
        *fix=0* to fix the first atom, *fix=1* to fix the second
        atom and *fix=0.5* (default) to fix the center of the bond.

        If *mask* or *indices* are set (*mask* overwrites *indices*),
        only the atoms defined there are moved
        (see :meth:`ase.Atoms.set_dihedral`).

        When *add* is true, the distance is changed by the value given.
        In combination
        with *factor* True, the value given is a factor scaling the distance.

        It is assumed that the atoms in *mask*/*indices* move together
        with *a1*. If *fix=1*, only *a0* will therefore be moved."""
    if a0 % len(self) == a1 % len(self):
        raise ValueError('a0 and a1 must not be the same')
    if add:
        oldDist = self.get_distance(a0, a1, mic=mic)
        if factor:
            newDist = oldDist * distance
        else:
            newDist = oldDist + distance
        self.set_distance(a0, a1, newDist, fix=fix, mic=mic, mask=mask, indices=indices, add=False, factor=False)
        return
    R = self.arrays['positions']
    D = np.array([R[a1] - R[a0]])
    if mic:
        D, D_len = find_mic(D, self.cell, self.pbc)
    else:
        D_len = np.array([np.sqrt((D ** 2).sum())])
    x = 1.0 - distance / D_len[0]
    if mask is None and indices is None:
        indices = [a0, a1]
    elif mask:
        indices = [i for i in range(len(self)) if mask[i]]
    for i in indices:
        if i == a0:
            R[a0] += x * fix * D[0]
        else:
            R[i] -= x * (1.0 - fix) * D[0]
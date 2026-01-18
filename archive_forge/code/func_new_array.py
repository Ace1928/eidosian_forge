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
def new_array(self, name, a, dtype=None, shape=None):
    """Add new array.

        If *shape* is not *None*, the shape of *a* will be checked."""
    if dtype is not None:
        a = np.array(a, dtype, order='C')
        if len(a) == 0 and shape is not None:
            a.shape = (-1,) + shape
    elif not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    else:
        a = a.copy()
    if name in self.arrays:
        raise RuntimeError('Array {} already present'.format(name))
    for b in self.arrays.values():
        if len(a) != len(b):
            raise ValueError('Array "%s" has wrong length: %d != %d.' % (name, len(a), len(b)))
        break
    if shape is not None and a.shape[1:] != shape:
        raise ValueError('Array "%s" has wrong shape %s != %s.' % (name, a.shape, a.shape[0:1] + shape))
    self.arrays[name] = a
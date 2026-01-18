import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
def add_morse(self, morse, atoms, row, col, data, conn=None):
    if self.hessian == 'reduced':
        i, j, Hx = ff.get_morse_potential_reduced_hessian(atoms, morse)
    elif self.hessian == 'spectral':
        i, j, Hx = ff.get_morse_potential_hessian(atoms, morse, spectral=True)
    else:
        raise NotImplementedError('Not implemented hessian')
    x = ij_to_x(i, j)
    row.extend(np.repeat(x, 6))
    col.extend(np.tile(x, 6))
    data.extend(Hx.flatten())
    if conn is not None:
        conn[i, j] = True
        conn[j, i] = True
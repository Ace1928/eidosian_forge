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
@property
def spline(self):
    s, x = self.get_coordinates()
    if self._spline and (np.abs(s - self._old_s).max() < 1e-06 and np.abs(x - self._old_x).max() < 1e-06):
        return self._spline
    self._spline = self.spline_fit()
    self._old_s = s
    self._old_x = x
    return self._spline
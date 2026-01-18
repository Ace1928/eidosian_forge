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
def get_tangent(self, i):
    """Normalised tangent vector at image i

        Args:
            i (int): image of interest

        Returns:
            tangent: tangent vector, normalised with appropriate precon norm
        """
    tangent = self.spline.dx_ds(self.spline.s[i])
    tangent /= self.precon[i].norm(tangent)
    return tangent.reshape(-1, 3)
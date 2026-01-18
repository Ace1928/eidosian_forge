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
def get_spring_force(self, i, k1, k2, tangent):
    """Spring force on image

        Args:
            i (int): image of interest
            k1 (float): spring constant for left spring
            k2 (float): spring constant for right spring
            tangent (array): tangent vector, shape (natoms, 3)

        Returns:
            eta: NEB spring forces, shape (natoms, 3)
        """
    nimages = len(self.images)
    k = 0.5 * (k1 + k2) / nimages ** 2
    curvature = self.spline.d2x_ds2(self.spline.s[i]).reshape(-1, 3)
    eta = k * self.precon[i].vdot(curvature, tangent) * tangent
    return eta
from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
@staticmethod
def get_principal_axis(coords, weights):
    """Get the molecule's principal axis.

        Args:
            coords: coordinates of atoms
            weights: the weight use for calculating the inertia tensor

        Returns:
            Array of dim 3 containing the principal axis
        """
    Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0
    for (x, y, z), wt in zip(coords, weights):
        Ixx += wt * (y * y + z * z)
        Iyy += wt * (x * x + z * z)
        Izz += wt * (x * x + y * y)
        Ixy += -wt * x * y
        Ixz += -wt * x * z
        Iyz += -wt * y * z
    inertia_tensor = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    _eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    return eigvecs[:, 0]
from __future__ import annotations
import collections
import itertools
import math
import operator
import warnings
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.due import Doi, due
def is_hexagonal(self, hex_angle_tol: float=5, hex_length_tol: float=0.01) -> bool:
    """
        Args:
            hex_angle_tol: Angle tolerance
            hex_length_tol: Length tolerance.

        Returns:
            Whether lattice corresponds to hexagonal lattice.
        """
    lengths = self.lengths
    angles = self.angles
    right_angles = [i for i in range(3) if abs(angles[i] - 90) < hex_angle_tol]
    hex_angles = [idx for idx in range(3) if abs(angles[idx] - 60) < hex_angle_tol or abs(angles[idx] - 120) < hex_angle_tol]
    return len(right_angles) == 2 and len(hex_angles) == 1 and (abs(lengths[right_angles[0]] - lengths[right_angles[1]]) < hex_length_tol)
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
def get_fractional_coords(self, cart_coords: ArrayLike) -> np.ndarray:
    """Returns the fractional coordinates given Cartesian coordinates.

        Args:
            cart_coords (3x1 array): Cartesian coords.

        Returns:
            Fractional coordinates.
        """
    return np.dot(cart_coords, self.inv_matrix)
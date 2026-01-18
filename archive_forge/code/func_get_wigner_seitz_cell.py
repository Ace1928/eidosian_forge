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
def get_wigner_seitz_cell(self) -> list[list[np.ndarray]]:
    """Returns the Wigner-Seitz cell for the given lattice.

        Returns:
            A list of list of coordinates.
            Each element in the list is a "facet" of the boundary of the
            Wigner Seitz cell. For instance, a list of four coordinates will
            represent a square facet.
        """
    vec1, vec2, vec3 = self._matrix
    list_k_points = []
    for ii, jj, kk in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
        list_k_points.append(ii * vec1 + jj * vec2 + kk * vec3)
    tess = Voronoi(list_k_points)
    out = []
    for r in tess.ridge_dict:
        if r[0] == 13 or r[1] == 13:
            out.append([tess.vertices[i] for i in tess.ridge_dict[r]])
    return out
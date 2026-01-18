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
def get_brillouin_zone(self) -> list[list[np.ndarray]]:
    """Returns the Wigner-Seitz cell for the reciprocal lattice, aka the
        Brillouin Zone.

        Returns:
            A list of list of coordinates.
            Each element in the list is a "facet" of the boundary of the
            Brillouin Zone. For instance, a list of four coordinates will
            represent a square facet.
        """
    return self.reciprocal_lattice.get_wigner_seitz_cell()
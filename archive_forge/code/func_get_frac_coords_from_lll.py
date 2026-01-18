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
def get_frac_coords_from_lll(self, lll_frac_coords: ArrayLike) -> np.ndarray:
    """Given fractional coordinates in the lll basis, returns corresponding
        fractional coordinates in the lattice basis.
        """
    return np.dot(lll_frac_coords, self.lll_mapping)
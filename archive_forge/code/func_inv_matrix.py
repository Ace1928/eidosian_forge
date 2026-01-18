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
@property
def inv_matrix(self) -> np.ndarray:
    """Inverse of lattice matrix."""
    if self._inv_matrix is None:
        self._inv_matrix = np.linalg.inv(self._matrix)
        self._inv_matrix.setflags(write=False)
    return self._inv_matrix
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
def lll_mapping(self) -> np.ndarray:
    """The mapping between the LLL reduced lattice and the original lattice."""
    if 0.75 not in self._lll_matrix_mappings:
        self._lll_matrix_mappings[0.75] = self._calculate_lll()
    return self._lll_matrix_mappings[0.75][1]
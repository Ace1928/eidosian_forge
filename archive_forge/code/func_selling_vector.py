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
def selling_vector(self) -> np.ndarray:
    """Returns the (1,6) array of Selling Scalars."""
    a, b, c = self.matrix
    d = -(a + b + c)
    tol = 1e-10
    selling_vector = np.array([np.dot(b, c), np.dot(a, c), np.dot(a, b), np.dot(a, d), np.dot(b, d), np.dot(c, d)])
    selling_vector = np.array([s if abs(s) > tol else 0 for s in selling_vector])
    reduction_matrices = [[[-1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0], [-1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1]], [[1, 1, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0], [0, -1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]], [[1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, -1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, -1, 0, 0, 1]], [[1, 0, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1]], [[0, 0, 1, 0, 1, 0], [0, 1, 0, 0, -1, 0], [1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 1, 1]], [[0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, -1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, -1]]]
    while np.greater(np.max(selling_vector), 0):
        max_index = selling_vector.argmax()
        selling_vector = np.dot(reduction_matrices[max_index], selling_vector)
    return selling_vector
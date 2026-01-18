from __future__ import annotations
import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
import numpy as np
import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
from ._qmc_cy import (
def in_neighborhood(candidate: np.ndarray, n: int=2) -> bool:
    """
            Check if there are samples closer than ``radius_squared`` to the
            `candidate` sample.
            """
    indices = (candidate / self.cell_size).astype(int)
    ind_min = np.maximum(indices - n, np.zeros(self.d, dtype=int))
    ind_max = np.minimum(indices + n + 1, self.grid_size)
    if not np.isnan(self.sample_grid[tuple(indices)][0]):
        return True
    a = [slice(ind_min[i], ind_max[i]) for i in range(self.d)]
    with np.errstate(invalid='ignore'):
        if np.any(np.sum(np.square(candidate - self.sample_grid[tuple(a)]), axis=self.d) < self.radius_squared):
            return True
    return False
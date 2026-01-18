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
def add_sample(candidate: np.ndarray) -> None:
    self.sample_pool.append(candidate)
    indices = (candidate / self.cell_size).astype(int)
    self.sample_grid[tuple(indices)] = candidate
    curr_sample.append(candidate)
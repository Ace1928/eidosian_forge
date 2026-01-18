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
def fast_forward(self, n: IntNumber) -> Sobol:
    """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : Sobol
            The fast-forwarded engine.

        """
    if self.num_generated == 0:
        _fast_forward(n=n - 1, num_gen=self.num_generated, dim=self.d, sv=self._sv, quasi=self._quasi)
    else:
        _fast_forward(n=n, num_gen=self.num_generated - 1, dim=self.d, sv=self._sv, quasi=self._quasi)
    self.num_generated += n
    return self
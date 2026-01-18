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
def random_base2(self, m: IntNumber) -> np.ndarray:
    """Draw point(s) from the Sobol' sequence.

        This function draws :math:`n=2^m` points in the parameter space
        ensuring the balance properties of the sequence.

        Parameters
        ----------
        m : int
            Logarithm in base 2 of the number of samples; i.e., n = 2^m.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
    n = 2 ** m
    total_n = self.num_generated + n
    if not total_n & total_n - 1 == 0:
        raise ValueError("The balance properties of Sobol' points require n to be a power of 2. {0} points have been previously generated, then: n={0}+2**{1}={2}. If you still want to do this, the function 'Sobol.random()' can be used.".format(self.num_generated, m, total_n))
    return self.random(n)
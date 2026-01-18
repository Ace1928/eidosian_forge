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
def integers(self, l_bounds: npt.ArrayLike, *, u_bounds: npt.ArrayLike | None=None, n: IntNumber=1, endpoint: bool=False, workers: IntNumber=1) -> np.ndarray:
    """
        Draw `n` integers from `l_bounds` (inclusive) to `u_bounds`
        (exclusive), or if endpoint=True, `l_bounds` (inclusive) to
        `u_bounds` (inclusive).

        Parameters
        ----------
        l_bounds : int or array-like of ints
            Lowest (signed) integers to be drawn (unless ``u_bounds=None``,
            in which case this parameter is 0 and this value is used for
            `u_bounds`).
        u_bounds : int or array-like of ints, optional
            If provided, one above the largest (signed) integer to be drawn
            (see above for behavior if ``u_bounds=None``).
            If array-like, must contain integer values.
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        endpoint : bool, optional
            If true, sample from the interval ``[l_bounds, u_bounds]`` instead
            of the default ``[l_bounds, u_bounds)``. Defaults is False.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Only supported when using `Halton`
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        Notes
        -----
        It is safe to just use the same ``[0, 1)`` to integer mapping
        with QMC that you would use with MC. You still get unbiasedness,
        a strong law of large numbers, an asymptotically infinite variance
        reduction and a finite sample variance bound.

        To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
        with :math:`a` the lower bounds and :math:`b` the upper bounds,
        the following transformation is used:

        .. math::

            \\text{floor}((b - a) \\cdot \\text{sample} + a)

        """
    if u_bounds is None:
        u_bounds = l_bounds
        l_bounds = 0
    u_bounds = np.atleast_1d(u_bounds)
    l_bounds = np.atleast_1d(l_bounds)
    if endpoint:
        u_bounds = u_bounds + 1
    if not np.issubdtype(l_bounds.dtype, np.integer) or not np.issubdtype(u_bounds.dtype, np.integer):
        message = "'u_bounds' and 'l_bounds' must be integers or array-like of integers"
        raise ValueError(message)
    if isinstance(self, Halton):
        sample = self.random(n=n, workers=workers)
    else:
        sample = self.random(n=n)
    sample = scale(sample, l_bounds=l_bounds, u_bounds=u_bounds)
    sample = np.floor(sample).astype(np.int64)
    return sample
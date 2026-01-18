from collections import namedtuple
import time
import logging
import warnings
import sys
import numpy as np
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper
from scipy.optimize._shgo_lib._complex import Complex
def g_topograph(self, x_min, X_min):
    """
        Returns the topographical vector stemming from the specified value
        ``x_min`` for the current feasible set ``X_min`` with True boolean
        values indicating positive entries and False values indicating
        negative entries.

        """
    x_min = np.array([x_min])
    self.Y = spatial.distance.cdist(x_min, X_min, 'euclidean')
    self.Z = np.argsort(self.Y, axis=-1)
    self.Ss = X_min[self.Z][0]
    self.minimizer_pool = self.minimizer_pool[self.Z]
    self.minimizer_pool = self.minimizer_pool[0]
    return self.Ss
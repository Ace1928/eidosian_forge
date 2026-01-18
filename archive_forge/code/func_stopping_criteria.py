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
def stopping_criteria(self):
    """
        Various stopping criteria ran every iteration

        Returns
        -------
        stop : bool
        """
    if self.maxiter is not None:
        self.finite_iterations()
    if self.iters is not None:
        self.finite_iterations()
    if self.maxfev is not None:
        self.finite_fev()
    if self.maxev is not None:
        self.finite_ev()
    if self.maxtime is not None:
        self.finite_time()
    if self.f_min_true is not None:
        self.finite_precision()
    if self.minhgrd is not None:
        self.finite_homology_growth()
    return self.stop_global
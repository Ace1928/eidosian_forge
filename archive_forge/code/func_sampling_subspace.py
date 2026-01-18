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
def sampling_subspace(self):
    """Find subspace of feasible points from g_func definition"""
    for ind, g in enumerate(self.g_cons):
        feasible = np.array([np.all(g(x_C, *self.g_args[ind]) >= 0.0) for x_C in self.C], dtype=bool)
        self.C = self.C[feasible]
        if self.C.size == 0:
            self.res.message = 'No sampling point found within the ' + 'feasible set. Increasing sampling ' + 'size.'
            if self.disp:
                logging.info(self.res.message)
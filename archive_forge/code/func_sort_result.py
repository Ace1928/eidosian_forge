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
def sort_result(self):
    """
        Sort results and build the global return object
        """
    results = self.LMC.sort_cache_result()
    self.res.xl = results['xl']
    self.res.funl = results['funl']
    self.res.x = results['x']
    self.res.fun = results['fun']
    self.res.nfev = self.fn + self.res.nlfev
    return self.res
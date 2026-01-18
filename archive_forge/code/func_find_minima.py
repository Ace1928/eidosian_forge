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
def find_minima(self):
    """
        Construct the minimizer pool, map the minimizers to local minima
        and sort the results into a global return object.
        """
    if self.disp:
        logging.info('Searching for minimizer pool...')
    self.minimizers()
    if len(self.X_min) != 0:
        self.minimise_pool(self.local_iter)
        self.sort_result()
        self.f_lowest = self.res.fun
        self.x_lowest = self.res.x
    else:
        self.find_lowest_vertex()
    if self.disp:
        logging.info(f'Minimiser pool = SHGO.X_min = {self.X_min}')
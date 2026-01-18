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
def finite_iterations(self):
    mi = min((x for x in [self.iters, self.maxiter] if x is not None))
    if self.disp:
        logging.info(f'Iterations done = {self.iters_done} / {mi}')
    if self.iters is not None:
        if self.iters_done >= self.iters:
            self.stop_global = True
    if self.maxiter is not None:
        if self.iters_done >= self.maxiter:
            self.stop_global = True
    return self.stop_global
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
def iterate_all(self):
    """
        Construct for `iters` iterations.

        If uniform sampling is used, every iteration adds 'n' sampling points.

        Iterations if a stopping criteria (e.g., sampling points or
        processing time) has been met.

        """
    if self.disp:
        logging.info('Splitting first generation')
    while not self.stop_global:
        if self.break_routine:
            break
        self.iterate()
        self.stopping_criteria()
    if not self.minimize_every_iter:
        if not self.break_routine:
            self.find_minima()
    self.res.nit = self.iters_done
    self.fn = self.HC.V.nfev
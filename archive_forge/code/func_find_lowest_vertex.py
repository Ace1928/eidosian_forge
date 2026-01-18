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
def find_lowest_vertex(self):
    self.f_lowest = np.inf
    for x in self.HC.V.cache:
        if self.HC.V[x].f < self.f_lowest:
            if self.disp:
                logging.info(f'self.HC.V[x].f = {self.HC.V[x].f}')
            self.f_lowest = self.HC.V[x].f
            self.x_lowest = self.HC.V[x].x_a
    for lmc in self.LMC.cache:
        if self.LMC[lmc].f_min < self.f_lowest:
            self.f_lowest = self.LMC[lmc].f_min
            self.x_lowest = self.LMC[lmc].x_l
    if self.f_lowest == np.inf:
        self.f_lowest = None
        self.x_lowest = None
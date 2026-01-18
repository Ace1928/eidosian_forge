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
def add_res(self, v, lres, bounds=None):
    v = np.ndarray.tolist(v)
    v = tuple(v)
    self.cache[v].x_l = lres.x
    self.cache[v].lres = lres
    self.cache[v].f_min = lres.fun
    self.cache[v].lbounds = bounds
    self.size += 1
    self.v_maps.append(v)
    self.xl_maps.append(lres.x)
    self.xl_maps_set.add(tuple(lres.x))
    self.f_maps.append(lres.fun)
    self.lbound_maps.append(bounds)
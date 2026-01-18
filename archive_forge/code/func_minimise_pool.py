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
def minimise_pool(self, force_iter=False):
    """
        This processing method can optionally minimise only the best candidate
        solutions in the minimiser pool

        Parameters
        ----------
        force_iter : int
                     Number of starting minimizers to process (can be specified
                     globally or locally)

        """
    lres_f_min = self.minimize(self.X_min[0], ind=self.minimizer_pool[0])
    self.trim_min_pool(0)
    while not self.stop_l_iter:
        self.stopping_criteria()
        if force_iter:
            force_iter -= 1
            if force_iter == 0:
                self.stop_l_iter = True
                break
        if np.shape(self.X_min)[0] == 0:
            self.stop_l_iter = True
            break
        self.g_topograph(lres_f_min.x, self.X_min)
        ind_xmin_l = self.Z[:, -1]
        lres_f_min = self.minimize(self.Ss[-1, :], self.minimizer_pool[-1])
        self.trim_min_pool(ind_xmin_l)
    self.stop_l_iter = False
    return
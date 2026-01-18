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
def iterate_delaunay(self):
    """
        Build a complex of Delaunay triangulated points

        Note: called with ``self.iterate_complex()`` after class initiation
        """
    self.nc += self.n
    self.sampled_surface(infty_cons_sampl=self.infty_cons_sampl)
    if self.disp:
        logging.info(f'self.n = {self.n}')
        logging.info(f'self.nc = {self.nc}')
        logging.info('Constructing and refining simplicial complex graph structure from sampling points.')
    if self.dim < 2:
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Ind_sorted = self.Ind_sorted.flatten()
        tris = []
        for ind, ind_s in enumerate(self.Ind_sorted):
            if ind > 0:
                tris.append(self.Ind_sorted[ind - 1:ind + 1])
        tris = np.array(tris)
        self.Tri = namedtuple('Tri', ['points', 'simplices'])(self.C, tris)
        self.points = {}
    else:
        if self.C.shape[0] > self.dim + 1:
            self.delaunay_triangulation(n_prc=self.n_prc)
        self.n_prc = self.C.shape[0]
    if self.disp:
        logging.info('Triangulation completed, evaluating all constraints and objective function values.')
    if hasattr(self, 'Tri'):
        self.HC.vf_to_vv(self.Tri.points, self.Tri.simplices)
    if self.disp:
        logging.info('Triangulation completed, evaluating all constraints and objective function values.')
    self.HC.V.process_pools()
    if self.disp:
        logging.info('Evaluations completed.')
    self.fn = self.HC.V.nfev
    self.n_sampled = self.nc
    return
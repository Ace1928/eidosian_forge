from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def print_setup_header(self, data, settings):
    """Print solver header
        """
    print('--------------------------------------------------------------')
    print('         OSQP v%s  -  Operator Splitting QP Solver' % self.version)
    print('                 Pure Python Implementation')
    print('        (c) Bartolomeo Stellato, Goran Banjac')
    print('      University of Oxford  -  Stanford University 2017')
    print('--------------------------------------------------------------')
    print('problem:  variables n = %d, constraints m = %d' % (data.n, data.m))
    nnz = self.work.data.P.nnz + self.work.data.A.nnz
    print('          nnz(P) + nnz(A) = %i' % nnz)
    print('settings: ', end='')
    if settings.linsys_solver == QDLDL_SOLVER:
        print('linear system solver = qdldl\n          ', end='')
    print('eps_abs = %.2e, eps_rel = %.2e,' % (settings.eps_abs, settings.eps_rel))
    print('          eps_prim_inf = %.2e, eps_dual_inf = %.2e,' % (settings.eps_prim_inf, settings.eps_dual_inf))
    print('          rho = %.2e ' % settings.rho, end='')
    if settings.adaptive_rho:
        print('(adaptive)')
    else:
        print('')
    print('          sigma = %.2e, alpha = %.2f, ' % (settings.sigma, settings.alpha), end='')
    print('max_iter = %d' % settings.max_iter)
    if settings.scaling:
        print('          scaling: on, ', end='')
    else:
        print('          scaling: off, ', end='')
    if settings.scaled_termination:
        print('scaled_termination: on')
    else:
        print('scaled_termination: off')
    if settings.warm_start:
        print('          warm_start: on, ', end='')
    else:
        print('          warm_start: off, ', end='')
    if settings.polish:
        print('polish: on')
    else:
        print('polish: off')
    print('')
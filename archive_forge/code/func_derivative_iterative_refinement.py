from __future__ import print_function
from builtins import object
import osqp._osqp as _osqp  # Internal low level module
import numpy as np
import scipy.sparse as spa
from warnings import warn
from platform import system
import osqp.codegen as cg
import osqp.utils as utils
import sys
import qdldl
def derivative_iterative_refinement(self, rhs, max_iter=20, tol=1e-12):
    M = self._derivative_cache['M']
    solver = self._derivative_cache['solver']
    sol = solver.solve(rhs)
    for k in range(max_iter):
        delta_sol = solver.solve(rhs - M @ sol)
        sol = sol + delta_sol
        if np.linalg.norm(M @ sol - rhs) < tol:
            break
    if k == max_iter - 1:
        warn('max_iter iterative refinement reached.')
    return sol
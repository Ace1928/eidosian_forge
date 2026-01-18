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
def adjoint_derivative(self, dx=None, dy_u=None, dy_l=None, P_idx=None, A_idx=None, eps_iter_ref=0.0001):
    """
        Compute adjoint derivative after solve.
        """
    P, q = (self._derivative_cache['P'], self._derivative_cache['q'])
    A = self._derivative_cache['A']
    l, u = (self._derivative_cache['l'], self._derivative_cache['u'])
    try:
        results = self._derivative_cache['results']
    except KeyError:
        raise ValueError('Problem has not been solved. You cannot take derivatives. Please call the solve function.')
    if results.info.status != 'solved':
        raise ValueError('Problem has not been solved to optimality. You cannot take derivatives')
    m, n = A.shape
    x = results.x
    y = results.y
    y_u = np.maximum(y, 0)
    y_l = -np.minimum(y, 0)
    if A_idx is None:
        A_idx = A.nonzero()
    if P_idx is None:
        P_idx = P.nonzero()
    if dy_u is None:
        dy_u = np.zeros(m)
    if dy_l is None:
        dy_l = np.zeros(m)
    if 'M' not in self._derivative_cache:
        inv_dia_y_u = spa.diags(np.reciprocal(y_u + 1e-20))
        inv_dia_y_l = spa.diags(np.reciprocal(y_l + 1e-20))
        M = spa.bmat([[P, A.T, -A.T], [A, spa.diags(A @ x - u) @ inv_dia_y_u, None], [-A, None, spa.diags(l - A @ x) @ inv_dia_y_l]], format='csc')
        delta = spa.bmat([[eps_iter_ref * spa.eye(n), None], [None, -eps_iter_ref * spa.eye(2 * m)]], format='csc')
        self._derivative_cache['M'] = M
        self._derivative_cache['solver'] = qdldl.Solver(M + delta)
    rhs = -np.concatenate([dx, dy_u, dy_l])
    r_sol = self.derivative_iterative_refinement(rhs)
    r_x, r_yu, r_yl = np.split(r_sol, [n, n + m])
    rows, cols = A_idx
    dA_vals = (y_u[rows] - y_l[rows]) * r_x[cols] + (r_yu[rows] - r_yl[rows]) * x[cols]
    dA = spa.csc_matrix((dA_vals, (rows, cols)), shape=A.shape)
    du = -r_yu
    dl = r_yl
    rows, cols = P_idx
    dP_vals = 0.5 * (r_x[rows] * x[cols] + r_x[cols] * x[rows])
    dP = spa.csc_matrix((dP_vals, P_idx), shape=P.shape)
    dq = r_x
    return (dP, dq, dA, dl, du)
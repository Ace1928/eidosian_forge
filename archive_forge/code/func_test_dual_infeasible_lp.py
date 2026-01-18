import osqp
from osqp._osqp import constant
import numpy as np
from scipy import sparse
import unittest
def test_dual_infeasible_lp(self):
    self.P = sparse.csc_matrix((2, 2))
    self.q = np.array([2, -1])
    self.A = sparse.eye(2, format='csc')
    self.l = np.array([0.0, 0.0])
    self.u = np.array([np.inf, np.inf])
    self.model = osqp.OSQP()
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.opts)
    res = self.model.solve()
    self.assertEqual(res.info.status_val, constant('OSQP_DUAL_INFEASIBLE'))
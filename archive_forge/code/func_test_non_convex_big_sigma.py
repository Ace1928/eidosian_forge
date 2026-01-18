import osqp
from osqp._osqp import constant
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_non_convex_big_sigma(self):
    opts = {'verbose': False, 'sigma': 5}
    self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **opts)
    res = self.model.solve()
    self.assertEqual(res.info.status_val, constant('OSQP_NON_CVX'))
    nptest.assert_approx_equal(res.info.obj_val, np.nan)
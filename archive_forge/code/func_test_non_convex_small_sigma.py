import osqp
from osqp._osqp import constant
import numpy as np
from scipy import sparse
import unittest
import numpy.testing as nptest
def test_non_convex_small_sigma(self):
    opts = {'verbose': False, 'sigma': 1e-06}
    try:
        test_setup = 1
        self.model.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **opts)
    except ValueError:
        test_setup = 0
    self.assertEqual(test_setup, 0)
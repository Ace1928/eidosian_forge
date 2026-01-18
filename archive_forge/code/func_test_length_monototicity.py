import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_length_monototicity(self) -> None:
    n = 5
    x = cp.Variable(n)
    self.assertTrue(cp.length(cp.abs(x)).is_incr(0))
    self.assertFalse(cp.length(cp.abs(x) - 1).is_incr(0))
    self.assertTrue(cp.length(cp.abs(x)).is_dqcp())
    self.assertFalse(cp.length(cp.abs(x) - 1).is_dqcp())
    self.assertTrue(cp.length(-cp.abs(x)).is_decr(0))
    self.assertFalse(cp.length(-cp.abs(x) + 1).is_decr(0))
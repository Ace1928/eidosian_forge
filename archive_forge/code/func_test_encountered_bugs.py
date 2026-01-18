import math
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.errors import InfeasibleConstraintException
import pyomo.contrib.fbbt.interval as interval
def test_encountered_bugs(self):
    lb, ub = self._inverse_power1(88893.4225, 88893.4225, 2, 2, 298.15, 298.15, 1e-08)
    self.assertAlmostEqual(lb, 298.15)
    self.assertAlmostEqual(ub, 298.15)
    lb, ub = self._inverse_power1(2.56e-06, 2.56e-06, 2, 2, -0.0016, -0.0016, 1e-12)
    self.assertAlmostEqual(lb, -0.0016)
    self.assertAlmostEqual(ub, -0.0016)
    lb, ub = self._inverse_power1(-1, -1e-12, 2, 2, -interval.inf, interval.inf, 1e-08)
    self.assertAlmostEqual(lb, 0)
    self.assertAlmostEqual(ub, 0)
    lb, ub = self.mul(0, 0, -interval.inf, interval.inf)
    self.assertEqual(lb, -interval.inf)
    self.assertEqual(ub, interval.inf)
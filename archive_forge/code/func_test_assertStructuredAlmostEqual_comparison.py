import datetime
import multiprocessing
import os
import time
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Param
def test_assertStructuredAlmostEqual_comparison(self):
    a = 1
    b = 1
    self.assertStructuredAlmostEqual(a, b)
    b -= 9.999e-08
    self.assertStructuredAlmostEqual(a, b)
    b -= 9.999e-08
    with self.assertRaisesRegex(self.failureException, '1 !~= 0.9999'):
        self.assertStructuredAlmostEqual(a, b)
    b = 1
    self.assertStructuredAlmostEqual(a, b, reltol=1e-06)
    b -= 9.999e-07
    self.assertStructuredAlmostEqual(a, b, reltol=1e-06)
    b -= 9.999e-07
    with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
        self.assertStructuredAlmostEqual(a, b, reltol=1e-06)
    b = 1
    self.assertStructuredAlmostEqual(a, b, places=6)
    b -= 9.999e-07
    self.assertStructuredAlmostEqual(a, b, places=6)
    b -= 9.999e-07
    with self.assertRaisesRegex(self.failureException, '1 !~= 0.999'):
        self.assertStructuredAlmostEqual(a, b, places=6)
    with self.assertRaisesRegex(self.failureException, '10 !~= 10.01'):
        self.assertStructuredAlmostEqual(10, 10.01, abstol=0.001)
    self.assertStructuredAlmostEqual(10, 10.01, reltol=0.001)
    with self.assertRaisesRegex(self.failureException, '10 !~= 10.01'):
        self.assertStructuredAlmostEqual(10, 10.01, delta=0.001)
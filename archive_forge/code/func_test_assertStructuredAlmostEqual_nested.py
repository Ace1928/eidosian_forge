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
def test_assertStructuredAlmostEqual_nested(self):
    a = {1.1: [1, 2, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
    b = {1.1: [1, 2, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
    self.assertStructuredAlmostEqual(a, b)
    b[1.1][2] -= 1.999e-07
    b[3][1] -= 9.999e-08
    self.assertStructuredAlmostEqual(a, b)
    b[1.1][2] -= 1.999e-07
    with self.assertRaisesRegex(self.failureException, '3 !~= 2.999'):
        self.assertStructuredAlmostEqual(a, b)
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
def test_assertStructuredAlmostEqual_numericvalue(self):
    m = ConcreteModel()
    m.v = Var(initialize=2.0)
    m.p = Param(initialize=2.0)
    a = {1.1: [1, m.p, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
    b = {1.1: [1, m.v, 3], 'a': 'hi', 3: {1: 2, 3: 4}}
    self.assertStructuredAlmostEqual(a, b)
    m.v.set_value(m.v.value - 1.999e-07)
    self.assertStructuredAlmostEqual(a, b)
    m.v.set_value(m.v.value - 1.999e-07)
    with self.assertRaisesRegex(self.failureException, '2.0 !~= 1.999'):
        self.assertStructuredAlmostEqual(a, b)
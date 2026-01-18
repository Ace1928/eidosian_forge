import os
from os.path import abspath, dirname
from io import StringIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
import random
from pyomo.opt import check_available_solvers
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.compare import assertExpressionsEqual
def checkUntransformedRule1(self, m, i):
    c = m.rule1[i]
    self.assertEqual(c.lower, 4)
    self.assertIsNone(c.upper)
    self.assertEqual(c.body.arg(0), 2)
    self.assertIs(c.body.arg(1), m.x[i])
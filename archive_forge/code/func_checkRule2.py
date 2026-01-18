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
def checkRule2(self, m):
    cons = m.rule2
    self.assertEqual(cons.upper, 6)
    self.assertIsNone(cons.lower)
    self.assertIs(cons.body, m.y)
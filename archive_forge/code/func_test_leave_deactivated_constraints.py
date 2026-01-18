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
def test_leave_deactivated_constraints(self):
    m = self.makeModel()
    m.rule2.deactivate()
    TransformationFactory('core.add_slack_variables').apply_to(m)
    cons = m.rule2
    self.assertFalse(cons.active)
    self.assertEqual(cons.lower, 1)
    self.assertEqual(cons.upper, 3)
    self.assertIs(cons.body, m.y)
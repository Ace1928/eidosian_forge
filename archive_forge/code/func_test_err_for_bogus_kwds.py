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
def test_err_for_bogus_kwds(self):
    m = self.makeModel()
    self.assertRaisesRegex(ValueError, "key 'notakwd' not defined for ConfigDict ''", TransformationFactory('core.add_slack_variables').apply_to, m, notakwd='I want a feasible model')
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
def checkConstraintDataObj(self, m):
    transBlock = m._core_add_slack_variables
    obj = transBlock.component('_slack_objective')
    self.assertIsInstance(obj, Objective)
    self.assertIs(obj.expr, transBlock.component('_slack_plus_rule1[2]'))
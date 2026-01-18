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
def checkSlackVars_indexedtarget(self, transBlock):
    self.assertIsInstance(transBlock.component('_slack_plus_rule1[1]'), Var)
    self.assertIsInstance(transBlock.component('_slack_plus_rule1[2]'), Var)
    self.assertIsInstance(transBlock.component('_slack_plus_rule1[3]'), Var)
    self.assertIsNone(transBlock.component('_slack_minus_rule2'))
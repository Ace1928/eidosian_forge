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
def checkTargetsObj(self, m):
    transBlock = m._core_add_slack_variables
    obj = transBlock.component('_slack_objective')
    assertExpressionsEqual(self, obj.expr, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, transBlock._slack_minus_rule1)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule3))]))
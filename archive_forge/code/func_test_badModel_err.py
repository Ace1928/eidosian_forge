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
def test_badModel_err(self):
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.rule1 = Constraint(expr=inequality(6, model.x, 5))
    self.assertRaisesRegex(RuntimeError, 'Lower bound exceeds upper bound in constraint rule1*', TransformationFactory('core.add_slack_variables').apply_to, model)
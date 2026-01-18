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
def test_deprecation_warning_for_cuid_target(self):
    m = self.makeModel()
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.core'):
        TransformationFactory('core.add_slack_variables').apply_to(m, targets=ComponentUID(m.rule3))
    self.assertRegex(out.getvalue(), 'DEPRECATED: In future releases ComponentUID targets will no longer be\nsupported in the core.add_slack_variables transformation. Specify\ntargets as a Constraint or list of Constraints.*')
    self.checkNonTargetCons(m)
    self.checkRule3(m)
    self.assertFalse(m.obj.active)
    self.checkTargetObj(m)
    transBlock = m.component('_core_add_slack_variables')
    self.checkTargetSlackVar(transBlock)
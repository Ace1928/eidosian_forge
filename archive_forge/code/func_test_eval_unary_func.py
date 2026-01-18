import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
def test_eval_unary_func(self):
    m = ConcreteModel()
    m.x = Var(initialize=4)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((log(m.x), None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, ('o43\n%s\n', [id(m.x)]))
    m.x.fix()
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((log(m.x), None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, math.log(4))
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
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
def test_linearexpression_npv(self):
    m = ConcreteModel()
    m.x = Var(initialize=4)
    m.y = Var(initialize=4)
    m.z = Var(initialize=4)
    m.p = Param(initialize=5, mutable=True)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((LinearExpression(args=[1, m.p, m.p * m.x, (m.p + 2) * m.y, 3 * m.z, m.p * m.z]), None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 6)
    self.assertEqual(repn.linear, {id(m.x): 5, id(m.y): 7, id(m.z): 8})
    self.assertEqual(repn.nonlinear, None)
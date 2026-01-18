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
def test_duplicate_shared_linear_expressions(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.e = Expression(expr=2 * m.x + 3 * m.y)
    expr1 = 10 * m.e
    expr2 = m.e + 100 * m.x + 100 * m.y
    info = INFO()
    with LoggingIntercept() as LOG:
        repn1 = info.visitor.walk_expression((expr1, None, None, 1))
        repn2 = info.visitor.walk_expression((expr2, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn1.nl, None)
    self.assertEqual(repn1.mult, 1)
    self.assertEqual(repn1.const, 0)
    self.assertEqual(repn1.linear, {id(m.x): 20, id(m.y): 30})
    self.assertEqual(repn1.nonlinear, None)
    self.assertEqual(repn2.nl, None)
    self.assertEqual(repn2.mult, 1)
    self.assertEqual(repn2.const, 0)
    self.assertEqual(repn2.linear, {id(m.x): 102, id(m.y): 103})
    self.assertEqual(repn2.nonlinear, None)
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
def test_eval_expr_if_lessEq(self):
    m = ConcreteModel()
    m.x = Var(initialize=4)
    m.y = Var(initialize=4)
    expr = Expr_if(m.x <= 4, m.x ** 2, m.y)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, ('o35\no23\n%s\nn4\no5\n%s\nn2\n%s\n', [id(m.x), id(m.x), id(m.y)]))
    m.x.fix()
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 16)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    m.x.fix(5)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {id(m.y): 1})
    self.assertEqual(repn.nonlinear, None)
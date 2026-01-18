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
def test_nested_operator_zero_arg(self):
    m = ConcreteModel()
    m.x = Var()
    m.p = Param(initialize=0, mutable=True)
    expr = 1 / m.x == m.p
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, ('o24\no3\nn1\n%s\nn0\n', [id(m.x)]))
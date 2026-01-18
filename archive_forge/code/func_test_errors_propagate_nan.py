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
def test_errors_propagate_nan(self):
    m = ConcreteModel()
    m.p = Param(mutable=True, initialize=0, domain=Any)
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.y.fix(1)
    expr = m.y ** 2 * m.x ** 2 * (3 * m.x / m.p * m.x) / m.y
    with LoggingIntercept() as LOG, INFO() as info:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(LOG.getvalue(), "Exception encountered evaluating expression 'div(3, 0)'\n\tmessage: division by zero\n\texpression: 3/p\n")
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    m.y.fix(None)
    expr = log(m.y) + 3
    with INFO() as info:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(str(repn.const), 'InvalidNumber(nan)')
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    expr = 3 * m.y
    with INFO() as info:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, InvalidNumber(None))
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    m.p.value = None
    expr = 5 * (m.p * m.x + 2 * m.z)
    with INFO() as info:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {id(m.z): 10, id(m.x): InvalidNumber(None)})
    self.assertEqual(repn.nonlinear, None)
    expr = m.y * m.x
    with INFO() as info:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {id(m.x): InvalidNumber(None)})
    self.assertEqual(repn.nonlinear, None)
    m.z = Var([1, 2, 3, 4], initialize=lambda m, i: i - 1)
    m.z[1].fix(None)
    expr = m.z[1] - m.z[2] * m.z[3] * m.z[4]
    with INFO() as info:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, InvalidNumber(None))
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear[0], 'o16\no2\no2\n%s\n%s\n%s\n')
    self.assertEqual(repn.nonlinear[1], [id(m.z[2]), id(m.z[3]), id(m.z[4])])
    m.z[3].fix(float('nan'))
    with INFO() as info:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, InvalidNumber(None))
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
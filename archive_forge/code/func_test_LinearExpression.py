import copy
import pickle
import math
import os
from collections import defaultdict
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.environ import (
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.expr_common import ExpressionType, clone_counter
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import expression_to_string, clone_expression
from pyomo.core.expr import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numvalue import NumericValue
def test_LinearExpression(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var([1, 2])
    e = LinearExpression()
    f = e.clone()
    self.assertIsNot(e, f)
    self.assertIsNot(e.linear_coefs, f.linear_coefs)
    self.assertIsNot(e.linear_vars, f.linear_vars)
    self.assertEqual(e.constant, f.constant)
    self.assertEqual(e.linear_coefs, f.linear_coefs)
    self.assertEqual(e.linear_vars, f.linear_vars)
    self.assertEqual(f.constant, 0)
    self.assertEqual(f.linear_coefs, [])
    self.assertEqual(f.linear_vars, [])
    e = LinearExpression(constant=5, linear_vars=[m.x, m.y[1]], linear_coefs=[10, 20])
    f = e.clone()
    self.assertIsNot(e, f)
    self.assertIsNot(e.linear_coefs, f.linear_coefs)
    self.assertIsNot(e.linear_vars, f.linear_vars)
    self.assertEqual(e.constant, f.constant)
    self.assertEqual(e.linear_coefs, f.linear_coefs)
    self.assertEqual(e.linear_vars, f.linear_vars)
    self.assertEqual(f.constant, 5)
    self.assertEqual(f.linear_coefs, [10, 20])
    self.assertEqual(f.linear_vars, [m.x, m.y[1]])
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
def test_nested_expr(self):
    expr1 = self.model.c * self.model.d
    expr2 = expr1 + expr1
    self.assertEqual(expr2.polynomial_degree(), 0)
    expr1 = self.model.a * self.model.b
    expr2 = expr1 + expr1
    self.assertEqual(expr2.polynomial_degree(), 2)
    self.model.a.fixed = True
    self.assertEqual(expr2.polynomial_degree(), 1)
    self.model.a.fixed = False
    expr1 = self.model.c + self.model.d
    expr2 = expr1 * expr1
    self.assertEqual(expr2.polynomial_degree(), 0)
    expr1 = self.model.a + self.model.b
    expr2 = expr1 * expr1
    self.assertEqual(expr2.polynomial_degree(), 2)
    self.model.a.fixed = True
    self.assertEqual(expr2.polynomial_degree(), 2)
    self.model.b.fixed = True
    self.assertEqual(expr2.polynomial_degree(), 0)
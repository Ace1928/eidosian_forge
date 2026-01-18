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
def test_Expr_if(self):
    m = self.instance
    expr = Expr_if(1, m.a, m.e)
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    m.a.fixed = True
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    m.a.fixed = False
    expr = Expr_if(0, m.a, m.e)
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), True)
    m.a.fixed = True
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), True)
    m.a.fixed = False
    expr = Expr_if(m.a, m.b, m.b)
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    m.a.fixed = True
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    m.a.fixed = False
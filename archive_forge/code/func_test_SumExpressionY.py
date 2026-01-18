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
def test_SumExpressionY(self):
    self.m = ConcreteModel()
    A = range(5)
    self.m.a = Var(A, initialize=5)
    self.m.b = Var(initialize=10)
    with clone_counter() as counter:
        start = counter.count
        expr1 = quicksum((self.m.a[i] for i in self.m.a))
        expr2 = copy.deepcopy(expr1)
        self.assertEqual(expr1(), 25)
        self.assertEqual(expr2(), 25)
        self.assertNotEqual(id(expr1), id(expr2))
        self.assertNotEqual(id(expr1._args_), id(expr2._args_))
        self.assertNotEqual(id(expr1.linear_vars[0]), id(expr2.linear_vars[0]))
        self.assertNotEqual(id(expr1.linear_vars[1]), id(expr2.linear_vars[1]))
        expr1 += self.m.b
        self.assertEqual(expr1(), 35)
        self.assertEqual(expr2(), 25)
        self.assertNotEqual(id(expr1), id(expr2))
        self.assertNotEqual(id(expr1._args_), id(expr2._args_))
        total = counter.count - start
        self.assertEqual(total, 0)
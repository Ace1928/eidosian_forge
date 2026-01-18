import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_expr4(self):
    model = AbstractModel()
    model.A = Set(initialize=[1, 2, 3])
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300}, mutable=True)
    model.x = Var(model.A)
    model.y = Var(model.A)
    instance = model.create_instance()
    expr = sum_product(denom=[instance.y, instance.x])
    baseline = '1/(y[1]*x[1]) + 1/(y[2]*x[2]) + 1/(y[3]*x[3])'
    self.assertEqual(str(expr), baseline)
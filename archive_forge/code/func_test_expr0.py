import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_expr0(self):
    model = AbstractModel()
    model.A = Set(initialize=[1, 2, 3])
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300}, mutable=True)
    model.C = Param(model.A, initialize={1: 100, 2: 200, 3: 300}, mutable=False)
    model.x = Var(model.A)
    model.y = Var(model.A)
    instance = model.create_instance()
    expr = sum_product(instance.B, instance.y)
    baseline = 'B[1]*y[1] + B[2]*y[2] + B[3]*y[3]'
    self.assertEqual(str(expr), baseline)
    expr = sum_product(instance.C, instance.y)
    self.assertEqual(str(expr), '100*y[1] + 200*y[2] + 300*y[3]')
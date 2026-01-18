import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_init_concrete(self):
    model = ConcreteModel()
    model.y = Var(initialize=0.0)
    model.x = Var(initialize=1.0)
    model.ec = Expression(expr=0)
    model.obj = Objective(expr=1.0 + model.ec)
    self.assertEqual(model.obj.expr(), 1.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e = 1.0
    model.ec.set_value(e)
    self.assertEqual(model.obj.expr(), 2.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e += model.x
    model.ec.set_value(e)
    self.assertEqual(model.obj.expr(), 3.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e += model.x
    self.assertEqual(model.obj.expr(), 3.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    model.del_component('obj')
    model.del_component('ec')
    model.ec = Expression(initialize=model.y)
    model.obj = Objective(expr=1.0 + model.ec)
    self.assertEqual(model.obj.expr(), 1.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e = 1.0
    model.ec.set_value(e)
    self.assertEqual(model.obj.expr(), 2.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e += model.x
    model.ec.set_value(e)
    self.assertEqual(model.obj.expr(), 3.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e += model.x
    self.assertEqual(model.obj.expr(), 3.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    model.del_component('obj')
    model.del_component('ec')
    model.y.set_value(-1)
    model.ec = Expression(initialize=model.y + 1.0)
    model.obj = Objective(expr=1.0 + model.ec)
    self.assertEqual(model.obj.expr(), 1.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e = 1.0
    model.ec.set_value(e)
    self.assertEqual(model.obj.expr(), 2.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e += model.x
    model.ec.set_value(e)
    self.assertEqual(model.obj.expr(), 3.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
    e += model.x
    self.assertEqual(model.obj.expr(), 3.0)
    self.assertEqual(id(model.obj.expr.arg(1)), id(model.ec))
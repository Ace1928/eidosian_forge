import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_init_abstract_indexed(self):
    model = AbstractModel()
    model.ec = Expression([1, 2, 3], initialize=1.0)
    model.obj = Objective(rule=lambda m: 1.0 + sum_product(m.ec, index=[1, 2, 3]))
    inst = model.create_instance()
    self.assertEqual(inst.obj.expr(), 4.0)
    inst.ec[1].set_value(2.0)
    self.assertEqual(inst.obj.expr(), 5.0)
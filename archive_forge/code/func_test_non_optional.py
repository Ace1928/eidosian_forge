import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_non_optional(self):
    m = ConcreteModel()
    m.i = IntervalVar(length=2, end=(4, 9), optional=False)
    self.assertEqual(value(m.i.is_present), True)
    self.assertTrue(m.i.is_present.fixed)
    self.assertFalse(m.i.optional)
    m.i2 = IntervalVar()
    self.assertEqual(value(m.i2.is_present), True)
    self.assertTrue(m.i.is_present.fixed)
    self.assertFalse(m.i2.optional)
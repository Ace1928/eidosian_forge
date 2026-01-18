import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_is_present_fixed_False(self):
    m = ConcreteModel()
    m.i = IntervalVar(optional=True)
    m.i.is_present.fix(False)
    self.assertTrue(m.i.optional)
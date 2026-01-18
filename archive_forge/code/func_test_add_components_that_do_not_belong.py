import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_add_components_that_do_not_belong(self):
    m = ConcreteModel()
    m.i = IntervalVar()
    with self.assertRaisesRegex(ValueError, 'Attempting to declare a block component using the name of a reserved attribute:\n\tnew_thing'):
        m.i.new_thing = IntervalVar()
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
def test_fixed_var_out_of_bounds_lb(self):
    m = ConcreteModel()
    m.s = RangeSet(2)
    m.v = Var(m.s, bounds=(0, 5))
    m.c = ConstraintList()
    m.c.add(expr=m.v[1] == m.v[2])
    m.c.add(expr=m.v[1] == -1)
    with self.assertRaises(ValueError):
        TransformationFactory('contrib.aggregate_vars').apply_to(m)
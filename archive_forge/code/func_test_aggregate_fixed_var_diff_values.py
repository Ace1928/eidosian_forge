import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
def test_aggregate_fixed_var_diff_values(self):
    m = ConcreteModel()
    m.s = RangeSet(3)
    m.v = Var(m.s, bounds=(0, 5))
    m.c = ConstraintList()
    m.c.add(expr=m.v[1] == m.v[2])
    m.c.add(expr=m.v[2] == m.v[3])
    m.c.add(expr=m.v[1] == 1)
    m.c.add(expr=m.v[3] == 3)
    with self.assertRaises(ValueError):
        TransformationFactory('contrib.aggregate_vars').apply_to(m)
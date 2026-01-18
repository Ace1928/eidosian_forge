from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_at_most(self):
    m = self.make_model()
    e = atmost(2, m.a, m.a.land(m.b), m.c)
    visitor = LogicalToDisjunctiveVisitor()
    m.cons = visitor.constraints
    m.z = visitor.z_vars
    m.disjuncts = visitor.disjuncts
    m.disjunctions = visitor.disjunctions
    visitor.walk_expression(e)
    self.assertIs(m.a.get_associated_binary(), m.z[1])
    a = m.z[1]
    self.assertIs(m.b.get_associated_binary(), m.z[2])
    b = m.z[2]
    self.assertIs(m.c.get_associated_binary(), m.z[4])
    c = m.z[4]
    self.assertEqual(len(m.z), 4)
    self.assertEqual(len(m.cons), 4)
    self.assertEqual(len(m.disjuncts), 2)
    self.assertEqual(len(m.disjunctions), 1)
    assertExpressionsEqual(self, m.cons[1].expr, m.z[3] <= a)
    assertExpressionsEqual(self, m.cons[2].expr, m.z[3] <= b)
    m.cons.pprint()
    print(m.cons[3].expr)
    assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[3] <= 2 - sum([a, b]))
    assertExpressionsEqual(self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[3] + m.z[4] <= 2)
    assertExpressionsEqual(self, m.disjuncts[1].constraint.expr, m.z[1] + m.z[3] + m.z[4] >= 3)
    assertExpressionsEqual(self, m.cons[4].expr, m.disjuncts[0].binary_indicator_var >= 1)
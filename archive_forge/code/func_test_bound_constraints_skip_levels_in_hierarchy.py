from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_bound_constraints_skip_levels_in_hierarchy(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.y = Var()
    m.Y = Disjunct([1, 2])
    m.Z = Disjunct([1, 2, 3])
    m.W = Disjunct([1, 2])
    m.W[1].c = Constraint(expr=m.x <= 7)
    m.W[2].c = Constraint(expr=m.x <= 9)
    m.Z[1].c = Constraint(expr=m.y == 0)
    m.Z[1].w_disj = Disjunction(expr=[m.W[i] for i in [1, 2]])
    m.Z[2].c = Constraint(expr=m.y == 1)
    m.Z[3].c = Constraint(expr=m.y == 2)
    m.Y[1].c = Constraint(expr=m.x >= 2)
    m.Y[1].z_disj = Disjunction(expr=[m.Z[i] for i in [1, 2, 3]])
    m.Y[2].c1 = Constraint(expr=m.x == 0)
    m.Y[2].c2 = Constraint(expr=(3, m.y, 17))
    m.y_disj = Disjunction(expr=[m.Y[i] for i in [1, 2]])
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.y_disj)
    self.assertEqual(len(cons), 2)
    x_lb = cons[0]
    assertExpressionsEqual(self, x_lb.expr, 2.0 * m.Y[1].binary_indicator_var + 0 * m.Y[2].binary_indicator_var <= m.x)
    x_ub = cons[1]
    assertExpressionsEqual(self, x_ub.expr, 10 * m.Y[1].binary_indicator_var + 0.0 * m.Y[2].binary_indicator_var >= m.x)
    self.assertIsNone(_parent_disjunct(x_lb))
    self.assertIsNone(_parent_disjunct(x_ub))
    cons = bt.get_transformed_constraints(m.y, m.y_disj)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.y, m.Y[1].z_disj)
    self.assertEqual(len(cons), 2)
    y_lb = cons[0]
    assertExpressionsEqual(self, y_lb.expr, 0.0 * m.Z[1].binary_indicator_var + 1.0 * m.Z[2].binary_indicator_var + 2.0 * m.Z[3].binary_indicator_var <= m.y)
    y_ub = cons[1]
    assertExpressionsEqual(self, y_ub.expr, 0.0 * m.Z[1].binary_indicator_var + 1.0 * m.Z[2].binary_indicator_var + 2.0 * m.Z[3].binary_indicator_var >= m.y)
    cons = bt.get_transformed_constraints(m.x, m.Y[1].z_disj)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.y, m.Z[1].w_disj)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.x, m.Z[1].w_disj)
    self.assertEqual(len(cons), 2)
    x_lb = cons[0]
    assertExpressionsEqual(self, x_lb.expr, 2.0 * m.W[1].binary_indicator_var + 2.0 * m.W[2].binary_indicator_var <= m.x)
    x_ub = cons[1]
    assertExpressionsEqual(self, x_ub.expr, 7.0 * m.W[1].binary_indicator_var + 9.0 * m.W[2].binary_indicator_var >= m.x)
    self.assertFalse(m.W[1].c.active)
    self.assertFalse(m.W[2].c.active)
    self.assertFalse(m.Z[1].c.active)
    self.assertFalse(m.Z[2].c.active)
    self.assertFalse(m.Z[3].c.active)
    self.assertFalse(m.Y[1].c.active)
    self.assertFalse(m.Y[2].c1.active)
    self.assertTrue(m.Y[2].c2.active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, descend_into=(Block, Disjunct), active=True))), 7)
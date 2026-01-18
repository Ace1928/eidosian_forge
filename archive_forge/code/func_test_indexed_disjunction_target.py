from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_indexed_disjunction_target(self):
    m = ConcreteModel()
    m.x = Var()
    m.d = Disjunct([1, 2, 3, 4, 5])
    m.d[1].c = Constraint(expr=m.x <= 1)
    m.d[2].c = Constraint(expr=m.x <= 2)
    m.d[3].c = Constraint(expr=m.x <= 3)
    m.d[4].c = Constraint(expr=m.x >= -5)
    m.d[5].c = Constraint(expr=m.x >= -8)
    m.disjunction = Disjunction(['pos', 'neg'])
    m.disjunction['pos'] = [m.d[1], m.d[2], m.d[3]]
    m.disjunction['neg'] = [m.d[4], m.d[5]]
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m, targets=m.disjunction)
    cons = bt.get_transformed_constraints(m.x, m.disjunction['pos'])
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    assertExpressionsEqual(self, ub.expr, 1.0 * m.d[1].binary_indicator_var + 2.0 * m.d[2].binary_indicator_var + 3.0 * m.d[3].binary_indicator_var >= m.x)
    cons = bt.get_transformed_constraints(m.x, m.disjunction['neg'])
    self.assertEqual(len(cons), 1)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, -5.0 * m.d[4].binary_indicator_var - 8.0 * m.d[5].binary_indicator_var <= m.x)
    self.assertEqual(len(list(m.component_data_objects(Constraint, descend_into=(Block, Disjunct), active=True))), 2)
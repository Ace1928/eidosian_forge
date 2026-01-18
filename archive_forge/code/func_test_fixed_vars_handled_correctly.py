from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_fixed_vars_handled_correctly(self):
    m = ConcreteModel()
    m.x = Var()
    m.x.setub(78)
    m.y = Var()
    m.y.fix(1)
    m.z = Var()
    m.disjunction = Disjunction(expr=[[m.x + m.y <= 5], [m.x <= 17], [m.z == 0]])
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.disjunction)
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    assertExpressionsEqual(self, ub.expr, 4.0 * m.disjunction.disjuncts[0].binary_indicator_var + 17.0 * m.disjunction.disjuncts[1].binary_indicator_var + 78 * m.disjunction.disjuncts[2].binary_indicator_var >= m.x)
    self.assertFalse(m.disjunction.disjuncts[0].constraint[1].active)
    self.assertFalse(m.disjunction.disjuncts[1].constraint[1].active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 2)